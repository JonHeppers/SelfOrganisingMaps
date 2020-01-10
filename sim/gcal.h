#include "morph/display.h"
#include "morph/tools.h"
#include <utility>
#include <iostream>
#include <unistd.h>
#include "morph/HexGrid.h"
#include "morph/ReadCurves.h"
#include "morph/RD_Base.h"
# include "morph/RD_Plot.h"

using namespace morph;
using namespace std;

using morph::RD_Plot;

template <class Flt>
class Projection{

/*
    A projection class for connecting units on a source sheet to units on a destination sheet with topographically aligned weighted connections from a radius of units on the source sheet to each destination sheet unit.
*/

public:

    HexGrid* hgSrc;
    HexGrid* hgDst;
    Flt radius;                    // radius within which connections are made
    Flt strength;                // strength of projection - multiplication after dot products
    Flt alpha;                    // learning rate
    unsigned int nSrc;                // number of units on source sheet
    unsigned int nDst;                // number of units on destination sheet

    vector<unsigned int> counts;                // number of connections in connection field for each unit
    vector<Flt> norms;                // 1./counts
    vector<Flt> alphas;                // learning rates for each unit may depend on e.g., the number of connections
    vector<vector<unsigned int> > srcId;            // identity of conneted units on the source sheet
    vector<vector<Flt> > weights;        // connection weights
    vector<vector<Flt> > distances;        // pre-compute distances between units in source and destination sheets
    vector<Flt> field;                // current activity patterns
    vector<Flt*> fSrc;                // pointers to the field elements on the source sheet
    vector<Flt*> fDst;                // pointers to the field elements on the destination sheet
    vector<double> weightPlot;            // for constructing activity plots
    bool normalizeAlphas;            // whether to normalize learning rate by individual unit connection density


    Projection(vector<Flt*> fSrc, vector<Flt*> fDst, HexGrid* hgSrc, HexGrid* hgDst, Flt radius, Flt strength, Flt alpha, Flt sigma, bool normalizeAlphas){

    /*
        Initialise the class with random weights (if sigma>0, the weights have a Gaussian pattern, else uniform random)
    */

        this->fSrc = fSrc;
        this->fDst = fDst;
        this->hgSrc = hgSrc;
        this->hgDst = hgDst;
        this->radius = radius;
        this->strength = strength;
        this->alpha = alpha;
    this->normalizeAlphas = normalizeAlphas;

        nDst = hgDst->vhexen.size();
        nSrc = hgSrc->vhexen.size();

        field.resize(nDst);
        counts.resize(nDst);
        norms.resize(nDst);
        srcId.resize(nDst);
        weights.resize(nDst);
        alphas.resize(nDst);
        distances.resize(nDst);
        weightPlot.resize(nSrc);

        Flt radiusSquared = radius*radius;    // precompute for speed

        double OverTwoSigmaSquared = 1./(sigma*sigma*2.0);    // precompute normalisation constant

    // initialize connections for each destination sheet unit
    #pragma omp parallel for
        for(unsigned int i=0;i<nDst;i++){
            for(unsigned int j=0;j<nSrc;j++){
                Flt dx = (hgSrc->vhexen[j]->x-hgDst->vhexen[i]->x);
                Flt dy = (hgSrc->vhexen[j]->y-hgDst->vhexen[i]->y);
                Flt distSquared = dx*dx+dy*dy;
                if (distSquared<radiusSquared){
                    counts[i]++;
                    srcId[i].push_back(j);
                    Flt w = 1.0;
                    if(sigma>0.){
                        w = exp(-distSquared*OverTwoSigmaSquared);
                    }
                    weights[i].push_back(w);
                    distances[i].push_back(sqrt(distSquared));
                }
            }
            norms[i] = 1.0/(Flt)counts[i];
            alphas[i] = alpha;
        if(normalizeAlphas){
        alphas[i] *= norms[i];
        }

    }
    }

    void getWeightedSum(void){
    /*
        Dot product of each weight vector with the corresponding source sheet field values, multiplied by the strength of the projection
    */
#pragma omp parallel for
        for(unsigned int i=0;i<nDst;i++){
            field[i] = 0.;
                for(unsigned int j=0;j<counts[i];j++){
                    field[i] += *fSrc[srcId[i][j]]*weights[i][j];
                }
        field[i] *= strength;
    }
    }

    void learn(void){
    /*
     Hebbian adaptation of the weights
    */
    if(alpha>0.0){
    #pragma omp parallel for
        for(unsigned int i=0;i<nDst;i++){
            for(unsigned int j=0;j<counts[i];j++){
                weights[i][j] += *fSrc[srcId[i][j]] * *fDst[i] * alphas[i];
            }
        }
    }
    }

    void renormalize(void){

    #pragma omp parallel for
        for(unsigned int i=0;i<nDst;i++){
        Flt sumWeights = 0.0;
            for(unsigned int j=0;j<counts[i];j++){
                sumWeights += weights[i][j];
            }
            for(unsigned int j=0;j<counts[i];j++){
                weights[i][j] /= sumWeights;
            }
        }
    }

    void multiplyWeights(int i, double scale){
    #pragma omp parallel for
        for(unsigned int j=0;j<counts[i];j++){
            weights[i][j] *= scale;
        }
    }

    vector<double> getWeightPlot(int i){

        #pragma omp parallel for
        for(unsigned int j=0;j<weightPlot.size();j++){
            weightPlot[j] = 0.;
        }
        #pragma omp parallel for
        for(unsigned int j=0;j<counts[i];j++){
            weightPlot[srcId[i][j]] = weights[i][j];
        }
        return weightPlot;
    }

};


template <class Flt>
class RD_Sheet : public morph::RD_Base<Flt>
{
public:

    vector<Projection<Flt>> Projections;
    alignas(alignof(vector<Flt>)) vector<Flt> X;
    alignas(alignof(vector<Flt*>)) vector<Flt*> Xptr;
    vector<vector<int> > P;            // identity of projections to (potentially) joint normalize

    virtual void init (void) {
        this->stepCount = 0;
        this->zero_vector_variable (this->X);
    }

    virtual void allocate (void) {
        morph::RD_Base<Flt>::allocate();
        this->resize_vector_variable (this->X);
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            Xptr.push_back(&this->X[hi]);
        }
    }

    void addProjection(vector<Flt*> inXptr, HexGrid* hgSrc, float radius, float strength, float alpha, float sigma, bool normalizeAlphas){
        Projections.push_back(Projection<Flt>(inXptr, this->Xptr, hgSrc, this->hg, radius, strength, alpha, sigma, normalizeAlphas));
    }

    void zero_X (void) {
        #pragma omp parallel for
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            this->X[hi] = 0.;
        }
    }

    void setNormalize(vector<int> proj){
    for(unsigned int p=0; p<proj.size();p++){
        for(unsigned int i=0; i<this->P.size();i++){
            for(unsigned int j=0; j<this->P[i].size();j++){
                if(proj[p]==this->P[i][j]){
                    cout<<"Caution - projection may be mutiply joint normalized"<<endl;
                }
            }
        }
    }
        this->P.push_back(proj);
    }

 void renormalize(void){

    for(unsigned int proj=0;proj<this->P.size();proj++){
    #pragma omp parallel for
        for(unsigned int i=0;i<this->nhex;i++){
            Flt sumWeights = 0.0;
            for(unsigned int p=0;p<this->P[proj].size();p++){
                for(unsigned int j=0;j<this->Projections[this->P[proj][p]].counts[i];j++){
                    sumWeights += this->Projections[this->P[proj][p]].weights[i][j];
                }
            }
            for(unsigned int p=0;p<this->P[proj].size();p++){
                for(unsigned int j=0;j<this->Projections[this->P[proj][p]].counts[i];j++){
                    this->Projections[this->P[proj][p]].weights[i][j] /= sumWeights;
                }
            }
        }
}
    }

};

template <class Flt>
class LGN : public RD_Sheet<Flt>
{

public:
Flt strength;

    LGN(Flt strength){
        this->strength = strength;
    }

    virtual void step (void) {
        this->stepCount++;
        this->zero_X();
        for(unsigned int i=0;i<this->Projections.size();i++){
            this->Projections[i].getWeightedSum();
        }

        for(unsigned int i=0;i<this->Projections.size();i++){
            #pragma omp parallel for
            for (unsigned int hi=0; hi<this->nhex; ++hi) {
                this->X[hi] += this->Projections[i].field[hi];
            }
        }
    #pragma omp parallel for
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            this->X[hi] = fmax(this->X[hi],0.);
        }
    }

};


template <class Flt>
class CortexSOM : public RD_Sheet<Flt>
{
    public:

    Flt beta, lambda, mu, oneMinusBeta, thetaInit;

    alignas(alignof(vector<Flt>)) vector<Flt> Xavg;
    alignas(alignof(vector<Flt>)) vector<Flt> Theta;


    CortexSOM(Flt beta, Flt lambda, Flt mu, Flt thetaInit){
        this->beta = beta;
        this->lambda = lambda;
        this->mu = mu;
    this->thetaInit = thetaInit;
        oneMinusBeta = (1.-beta);
    }

    virtual void init (void) {
        this->stepCount = 0;
        this->zero_vector_variable (this->X);
        this->zero_vector_variable (this->Xavg);
        this->zero_vector_variable (this->Theta);
    }

    virtual void allocate (void) {
        morph::RD_Base<Flt>::allocate();
        this->resize_vector_variable (this->X);
        this->resize_vector_variable (this->Xavg);
        this->resize_vector_variable (this->Theta);
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            this->Xptr.push_back(&this->X[hi]);
        }
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            this->Xavg[hi] = mu;
            this->Theta[hi] = thetaInit;
        }
    }


    virtual void step (void) {
        this->stepCount++;

        for(unsigned int i=0;i<this->Projections.size();i++){
            this->Projections[i].getWeightedSum();
        }

        this->zero_X();

        for(unsigned int i=0;i<this->Projections.size();i++){
        #pragma omp parallel for
            for (unsigned int hi=0; hi<this->nhex; ++hi) {
                this->X[hi] += this->Projections[i].field[hi];
            }
        }

        #pragma omp parallel for
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            this->X[hi] = this->X[hi]-this->Theta[hi];
            if(this->X[hi]<0.0){
                this->X[hi] = 0.0;
            }
        }
    }

    void homeostasis(void){

     #pragma omp parallel for
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            this->Xavg[hi] = oneMinusBeta*this->X[hi] + beta*this->Xavg[hi];
        }
     #pragma omp parallel for
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            this->Theta[hi] = this->Theta[hi] + lambda*(this->Xavg[hi]-mu);
        }
    }



};

template <class Flt>
class GaussianOriented_Sheet : public RD_Sheet<Flt>
{
public:
    GaussianOriented_Sheet(){ }

    virtual void step (void) {
        this->stepCount++;
    }

    void GeneratePattern(double x_center, double y_center, double theta, double sigmaA, double sigmaB){
        double cosTheta = cos(theta);
        double sinTheta = sin(theta);
        double overSigmaA = 1./sigmaA;
        double overSigmaB = 1./sigmaB;
        #pragma omp parallel for
        for (unsigned int hi=0; hi<this->nhex; ++hi) {
            Flt dx = this->hg->vhexen[hi]->x-x_center;
            Flt dy = this->hg->vhexen[hi]->y-y_center;
            this->X[hi] = exp(-((dx*cosTheta-dy*sinTheta)*(dx*cosTheta-dy*sinTheta))*overSigmaA
                              -((dx*sinTheta+dy*cosTheta)*(dx*sinTheta+dy*cosTheta))*overSigmaB);
        }
    }

};
