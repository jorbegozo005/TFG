extern int nplanetas;
extern int xyz;
extern int gmax;

void initialInnerPlanets(float *Gm, float *u, int tamano) {

/*
Order of planets:
Sun, Mercury, Venus, Earth, Mars, Moon
*/

    float GmMoon = 0.109318945074237400e-10;
    float GmEarth = 0.888769244512563400e-9;

    Gm[0] = 0.295912208285591100e-3;
    Gm[1] = 0.491248045036476000e-10;
    Gm[2] = 0.724345233264412000e-9;
    Gm[3] = GmEarth+GmMoon;
    Gm[4] = 0.954954869555077000e-10;

    float q0[15] = {0.00450250878464055477, 0.00076707642709100705, 0.00026605791776697764,   // Sun
        0.36176271656028195477, -0.09078197215676599295, -0.08571497256275117236, // Mercury
        0.61275194083507215477, -0.34836536903362219295, -0.19527828667594382236, // Venus
        0.12051741410138465477, -0.92583847476914859295, -0.40154022645315222236, // EM bary
        -0.11018607714879824523, -1.32759945030298299295,-0.60588914048429142236  // Mars
    };

    float v[15] = {-0.00000035174953607552, 0.00000517762640983341, 0.00000222910217891203,  // Sun
      0.00336749397200575848, 0.02489452055768343341, 0.01294630040970409203,  // Mercury
      0.01095206842352823448, 0.01561768426786768341, 0.00633110570297786403,  // Venus
      0.01681126830978379448, 0.00174830923073434441, 0.00075820289738312913,  // EM bary
      0.01448165305704756448, 0.00024246307683646861, -0.00028152072792433877  // Mars
    };

    float sumGm = 0.0;
    for (int i=0; i<5; i++) {
       sumGm = sumGm + Gm[i];
    }

    float resultadoqx = 0.0;
    float resultadoqy = 0.0;
    float resultadoqz = 0.0;
    for (int i=0; i<5; i++) {
       resultadoqx =  resultadoqx + (Gm[i]*q0[i*3]);
       resultadoqy =  resultadoqy + (Gm[i]*q0[i*3+1]);
       resultadoqz =  resultadoqz + (Gm[i]*q0[i*3+2]);
    }
    float qxbar = resultadoqx/sumGm;
    float qybar = resultadoqy/sumGm;
    float qzbar = resultadoqz/sumGm;

    float resultadovx = 0.0;
    float resultadovy = 0.0;
    float resultadovz = 0.0;
    for (int i=0; i<5; i++) {
       resultadovx =  resultadovx + (Gm[i]*v[i*3]);
       resultadovy =  resultadovy + (Gm[i]*v[i*3+1]);
       resultadovz =  resultadovz + (Gm[i]*v[i*3+2]);
    }
    float vxbar = resultadovx/sumGm;
    float vybar = resultadovy/sumGm;
    float vzbar = resultadovz/sumGm;

    for (int i=0; i<5; i++) {
       q0[i*3] = q0[i*3] - qxbar;
       q0[i*3+1] = q0[i*3+1] - qybar;
       q0[i*3+2] = q0[i*3+2] - qzbar;
    }

    for (int i=0; i<5; i++) {
       v[i*3] = v[i*3] - vxbar;
       v[i*3+1] = v[i*3+1] - vybar;
       v[i*3+2] = v[i*3+2] - vzbar;
    }

    for (int i=0; i<tamano; i++) {
        u[i] = q0[i];
        u[i+tamano] = v[i];
    }

    for (int i=nplanetas*xyz*2; i<gmax*nplanetas*xyz; i++) {
        u[i] = 0.0;
    }

}
