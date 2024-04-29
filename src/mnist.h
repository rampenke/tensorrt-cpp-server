#include "model.h"

class MnistApi: public Model {
public:
    virtual bool load();
    virtual int infer(const char *data);
public:
    void *mModel; 
};