struct NBodyParams {
    float *gm;
    float *aux;
};

void NbodyODE2Tp(float *ddq, float *q, struct NBodyParams *p);
