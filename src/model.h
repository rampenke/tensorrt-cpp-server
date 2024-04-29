
class Model {
public:
    virtual bool load() = 0;
    virtual int infer(const char *data) = 0;
};