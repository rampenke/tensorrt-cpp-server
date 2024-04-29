#include <memory>
#include <vector>
#include <NvInfer.h>
#include <numeric>
#include <unordered_map>
#include <assert.h>

#undef CHECK
#define CHECK(status)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cerr << "Cuda failure: " << ret << std::endl;                                                         \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#undef ASSERT
#define ASSERT(condition)                                                   \
    do                                                                      \
    {                                                                       \
        if (!(condition))                                                   \
        {                                                                   \
            exit(EXIT_FAILURE);                                                       \
        }                                                                   \
    } while (0)

// sample::gLogError << "Assertion failure: " << #condition << std::endl;

#undef SAFE_ASSERT
#define SAFE_ASSERT(condition)                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            std::cerr << "Assertion failure: " << #condition << std::endl;                                             \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)
    
template <typename A, typename B>
inline A divUp(A x, B n) {
    return (x + n - 1) / n;
}

inline int64_t volume(nvinfer1::Dims const& d) {
    return std::accumulate(d.d, d.d + d.nbDims, int64_t{1}, std::multiplies<int64_t>{});
}

inline uint32_t getElementSize(nvinfer1::DataType t) noexcept {
    switch (t) {
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kINT32:
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kBF16:
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kFP8: return 1;
    //case nvinfer1::DataType::kINT4: ASSERT(false && "Element size is not implemented for sub-byte data-types (INT4)");
    }
    return 0;
}

template <typename AllocFunc, typename FreeFunc>
class GenericBuffer {
public:
    //!
    //! \brief Construct an empty buffer.
    //!
    GenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
        : mSize(0)
        , mCapacity(0)
        , mType(type)
        , mBuffer(nullptr)
    {
    }

    //!
    //! \brief Construct a buffer with the specified allocation size in bytes.
    //!
    GenericBuffer(size_t size, nvinfer1::DataType type)
        : mSize(size)
        , mCapacity(size)
        , mType(type)
    {
        if (!allocFn(&mBuffer, this->nbBytes())) {
            throw std::bad_alloc();
        }
    }

    GenericBuffer(GenericBuffer&& buf)
        : mSize(buf.mSize)
        , mCapacity(buf.mCapacity)
        , mType(buf.mType)
        , mBuffer(buf.mBuffer)
    {
        buf.mSize = 0;
        buf.mCapacity = 0;
        buf.mType = nvinfer1::DataType::kFLOAT;
        buf.mBuffer = nullptr;
    }

    GenericBuffer& operator=(GenericBuffer&& buf)
    {
        if (this != &buf) {
            freeFn(mBuffer);
            mSize = buf.mSize;
            mCapacity = buf.mCapacity;
            mType = buf.mType;
            mBuffer = buf.mBuffer;
            // Reset buf.
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    void* data() {
        return mBuffer;
    }

    //!
    //! \brief Returns pointer to underlying array.
    //!
    const void* data() const {
        return mBuffer;
    }

    //!
    //! \brief Returns the size (in number of elements) of the buffer.
    //!
    size_t size() const {
        return mSize;
    }

    //!
    //! \brief Returns the size (in bytes) of the buffer.
    //!
    size_t nbBytes() const {
        return this->size() * getElementSize(mType);
    }

    //!
    //! \brief Resizes the buffer. This is a no-op if the new size is smaller than or equal to the current capacity.
    //!
    void resize(size_t newSize) {
        mSize = newSize;
        if (mCapacity < newSize) {
            freeFn(mBuffer);
            if (!allocFn(&mBuffer, this->nbBytes())) {
                throw std::bad_alloc{};
            }
            mCapacity = newSize;
        }
    }

    //!
    //! \brief Overload of resize that accepts Dims
    //!
    void resize(const nvinfer1::Dims& dims) {
        return this->resize(volume(dims));
    }

    ~GenericBuffer() {
        freeFn(mBuffer);
    }

private:
    size_t mSize{0}, mCapacity{0};
    nvinfer1::DataType mType;
    void* mBuffer;
    AllocFunc allocFn;
    FreeFunc freeFn;
};


class DeviceAllocator {
public:
    bool operator()(void** ptr, size_t size) const {
        return cudaMalloc(ptr, size) == cudaSuccess;
    }
};

class DeviceFree {
public:
    void operator()(void* ptr) const {
        cudaFree(ptr);
    }
};

class HostAllocator {
public:
    bool operator()(void** ptr, size_t size) const {
        *ptr = malloc(size);
        return *ptr != nullptr;
    }
};

class HostFree {
public:
    void operator()(void* ptr) const {
        free(ptr);
    }
};

using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

class ManagedBuffer {
public:
    DeviceBuffer deviceBuffer;
    HostBuffer hostBuffer;
};


class BufferManager {
public:
    static const size_t kINVALID_SIZE_VALUE = ~size_t(0);

    //!
    //! \brief Create a BufferManager for handling buffer interactions with engine.
    //!
    BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine, int32_t const batchSize = 0,
        nvinfer1::IExecutionContext const* context = nullptr)
        : mEngine(engine)
        , mBatchSize(batchSize) {
        // Create host and device buffers
        for (int32_t i = 0, e = mEngine->getNbIOTensors(); i < e; i++) {
            auto const name = engine->getIOTensorName(i);
            mNames[name] = i;

            auto dims = context ? context->getTensorShape(name) : mEngine->getTensorShape(name);
            size_t vol = context || !mBatchSize ? 1 : static_cast<size_t>(mBatchSize);
            nvinfer1::DataType type = mEngine->getTensorDataType(name);
            int32_t vecDim = mEngine->getTensorVectorizedDim(name);
            if (-1 != vecDim) { // i.e., 0 != lgScalarsPerVector 
                int32_t scalarsPerVec = mEngine->getTensorComponentsPerElement(name);
                dims.d[vecDim] = divUp(dims.d[vecDim], scalarsPerVec);
                vol *= scalarsPerVec;
            }
            vol *= volume(dims);
            std::unique_ptr<ManagedBuffer> manBuf{new ManagedBuffer()};
            manBuf->deviceBuffer = DeviceBuffer(vol, type);
            manBuf->hostBuffer = HostBuffer(vol, type);
            void* deviceBuffer = manBuf->deviceBuffer.data();
            mDeviceBindings.emplace_back(deviceBuffer);
            mManagedBuffers.emplace_back(std::move(manBuf));
        }
    }

    //!
    //! \brief Returns a vector of device buffers that you can use directly as
    //!        bindings for the execute and enqueue methods of IExecutionContext.
    //!
    std::vector<void*>& getDeviceBindings(){
        return mDeviceBindings;
    }

    //!
    //! \brief Returns a vector of device buffers.
    //!
    std::vector<void*> const& getDeviceBindings() const
    {
        return mDeviceBindings;
    }

    //!
    //! \brief Returns the device buffer corresponding to tensorName.
    //!        Returns nullptr if no such tensor can be found.
    //!
    void* getDeviceBuffer(std::string const& tensorName) const
    {
        return getBuffer(false, tensorName);
    }

    //!
    //! \brief Returns the host buffer corresponding to tensorName.
    //!        Returns nullptr if no such tensor can be found.
    //!
    void* getHostBuffer(std::string const& tensorName) const
    {
        return getBuffer(true, tensorName);
    }

    //!
    //! \brief Returns the size of the host and device buffers that correspond to tensorName.
    //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
    //!
    size_t size(std::string const& tensorName) const
    {
        auto record = mNames.find(tensorName);
        if (record == mNames.end())
            return kINVALID_SIZE_VALUE;
        return mManagedBuffers[record->second]->hostBuffer.nbBytes();
    }

    //!
    //! \brief Templated print function that dumps buffers of arbitrary type to std::ostream.
    //!        rowCount parameter controls how many elements are on each line.
    //!        A rowCount of 1 means that there is only 1 element on each line.
    //!
    template <typename T>
    void print(std::ostream& os, void* buf, size_t bufSize, size_t rowCount)
    {
        assert(rowCount != 0);
        assert(bufSize % sizeof(T) == 0);
        T* typedBuf = static_cast<T*>(buf);
        size_t numItems = bufSize / sizeof(T);
        for (int32_t i = 0; i < static_cast<int>(numItems); i++)
        {
            // Handle rowCount == 1 case
            if (rowCount == 1 && i != static_cast<int>(numItems) - 1)
                os << typedBuf[i] << std::endl;
            else if (rowCount == 1)
                os << typedBuf[i];
            // Handle rowCount > 1 case
            else if (i % rowCount == 0)
                os << typedBuf[i];
            else if (i % rowCount == rowCount - 1)
                os << " " << typedBuf[i] << std::endl;
            else
                os << " " << typedBuf[i];
        }
    }

    //!
    //! \brief Copy the contents of input host buffers to input device buffers synchronously.
    //!
    void copyInputToDevice()
    {
        memcpyBuffers(true, false, false);
    }

    //!
    //! \brief Copy the contents of output device buffers to output host buffers synchronously.
    //!
    void copyOutputToHost()
    {
        memcpyBuffers(false, true, false);
    }

    //!
    //! \brief Copy the contents of input host buffers to input device buffers asynchronously.
    //!
    void copyInputToDeviceAsync(cudaStream_t const& stream = 0)
    {
        memcpyBuffers(true, false, true, stream);
    }

    //!
    //! \brief Copy the contents of output device buffers to output host buffers asynchronously.
    //!
    void copyOutputToHostAsync(cudaStream_t const& stream = 0)
    {
        memcpyBuffers(false, true, true, stream);
    }

    ~BufferManager() = default;

private:
    void* getBuffer(bool const isHost, std::string const& tensorName) const
    {
        auto record = mNames.find(tensorName);
        if (record == mNames.end())
            return nullptr;
        return (isHost ? mManagedBuffers[record->second]->hostBuffer.data()
                       : mManagedBuffers[record->second]->deviceBuffer.data());
    }

    bool tenosrIsInput(const std::string& tensorName) const
    {
        return mEngine->getTensorIOMode(tensorName.c_str()) == nvinfer1::TensorIOMode::kINPUT;
    }

    void memcpyBuffers(bool const copyInput, bool const deviceToHost, bool const async, cudaStream_t const& stream = 0)
    {
        for (auto const& n : mNames)
        {
            void* dstPtr = deviceToHost ? mManagedBuffers[n.second]->hostBuffer.data()
                                        : mManagedBuffers[n.second]->deviceBuffer.data();
            void const* srcPtr = deviceToHost ? mManagedBuffers[n.second]->deviceBuffer.data()
                                              : mManagedBuffers[n.second]->hostBuffer.data();
            size_t const byteSize = mManagedBuffers[n.second]->hostBuffer.nbBytes();
            const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
            if ((copyInput && tenosrIsInput(n.first)) || (!copyInput && !tenosrIsInput(n.first)))
            {
                if (async)
                    CHECK(cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream));
                else
                    CHECK(cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType));
            }
        }
    }

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;              //!< The pointer to the engine
    int mBatchSize;                                              //!< The batch size for legacy networks, 0 otherwise.
    std::vector<std::unique_ptr<ManagedBuffer>> mManagedBuffers; //!< The vector of pointers to managed buffers
    std::vector<void*> mDeviceBindings;              //!< The vector of device buffers needed for engine execution
    std::unordered_map<std::string, int32_t> mNames; //!< The map of tensor name and index pairs
};