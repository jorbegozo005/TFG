float *TaylorLortuP(float *nu, float *nddu, struct NBodyParams *p);
void evaluatetaylorv(float *nu, float h);
void TaylorStepP(float *u, float *ddu, struct NBodyParams *p, float h);
void IntegrateTaylorP(float *u, int t0, int tf, float h, float *gm);
