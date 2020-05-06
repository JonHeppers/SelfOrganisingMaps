#include "opencv2/opencv.hpp"
#include "morph/display.h"
#include "morph/tools.h"
#include <utility>
#include <iostream>
#include <unistd.h>
#include "morph/HexGrid.h"
#include "morph/ReadCurves.h"
#include "morph/RD_Base.h"
#include "morph/RD_Plot.h"
#include "topo.h"
#include <ctime>
#include <chrono>
#include <string>

using namespace morph;
using namespace std;

/*** SPECIFIC MODEL DEFINED HERE ***/

class gcal: public Network {

public:

	HexCartSampler<double> HCM;
	PatternGenerator_Sheet<double> IN;
	LGN<double> LGN_ON, LGN_OFF;
	CortexSOM<double> CX;
	vector<double> orientationPref, orientationSel, directionPref, directionSel;
	bool homeostasis, plotSettling, zeroCortex, decayCortex;
	unsigned int settle;
	float beta, lambda, mu, thetaInit, xRange, yRange, afferAlpha, excitAlpha,
			inhibAlpha;
	float afferStrength, excitStrength, inhibStrength, LGNstrength, scale;
	float sigmaA, sigmaB, afferRadius, excitRadius, inhibRadius, afferSigma,
			excitSigma, inhibSigma, LGNCenterSigma, LGNSurroundSigma;
	// Grating parameters
	float gratingWidth, numGratingOrientations, numGratingPhases,
			numDirectionOrientations;

	gcal(void) {
		plotSettling = false;
	}

	void init(Json::Value root) {

		// GET PARAMS FROM JSON
		homeostasis = root.get("homeostasis", true).asBool();
		settle = root.get("settle", 16).asUInt();

		// homeostasis
		beta = root.get("beta", 0.991).asFloat();
		lambda = root.get("lambda", 0.01).asFloat();
		mu = root.get("thetaInit", 0.15).asFloat();
		thetaInit = root.get("mu", 0.024).asFloat();
		xRange = root.get("xRange", 2.0).asFloat();
		yRange = root.get("yRange", 2.0).asFloat();

		// learning rates
		afferAlpha = root.get("afferAlpha", 0.1).asFloat();
		excitAlpha = root.get("excitAlpha", 0.0).asFloat();
		inhibAlpha = root.get("inhibAlpha", 0.3).asFloat();

		// projection strengths
		afferStrength = root.get("afferStrength", 1.5).asFloat();
		excitStrength = root.get("excitStrength", 1.7).asFloat();
		inhibStrength = root.get("inhibStrength", -1.4).asFloat();
		LGNstrength = root.get("LGNstrength", 14.0).asFloat();

		// spatial params
		scale = root.get("scale", 0.5).asFloat();
		sigmaA = root.get("sigmaA", 1.0).asFloat() * scale;
		sigmaB = root.get("sigmaB", 0.3).asFloat() * scale;
		afferRadius = root.get("afferRadius", 0.27).asFloat() * scale;
		excitRadius = root.get("excitRadius", 0.1).asFloat() * scale;
		inhibRadius = root.get("inhibRadius", 0.23).asFloat() * scale;
		afferSigma = root.get("afferSigma", 0.270).asFloat() * scale;
		excitSigma = root.get("excitSigma", 0.025).asFloat() * scale;
		inhibSigma = root.get("inhibSigma", 0.075).asFloat() * scale;
		LGNCenterSigma = root.get("LGNCenterSigma", 0.037).asFloat() * scale;
		LGNSurroundSigma = root.get("LGNSuroundSigma", 0.150).asFloat() * scale;

		// sin grating parameters
		gratingWidth = root.get("gratingWidth", 30).asFloat();
		numGratingOrientations =
				root.get("numGratingOrientations", 20).asFloat();
		numGratingPhases = root.get("numGratingPhases", 8).asFloat();
		numDirectionOrientations = numGratingOrientations * 2;

		// Cortex zero/decay bools
		zeroCortex = root.get("zeroCortex", true).asBool();
		decayCortex = root.get("decayCortex", false).asBool();

		// INITIALIZE LOGFILE
		stringstream fname;
		string logpath = root.get("logpath", "logs/").asString();
		morph::Tools::createDir(logpath);
		fname << logpath << "/log.h5";
		HdfData data(fname.str());

		// Mapping Between Hexagonal and Cartesian Sheet
		HCM.svgpath =
				root.get("IN_svgpath", "boundaries/trialmod.svg").asString();
		HCM.init();
		HCM.allocate();

		// INPUT SHEET
		IN.svgpath =
				root.get("IN_svgpath", "boundaries/trialmod.svg").asString();
		IN.init();
		IN.allocate();

		// LGN ON CELLS
		LGN_ON.strength = LGNstrength;
		LGN_ON.svgpath =
				root.get("LGN_svgpath", "boundaries/trialmod.svg").asString();
		LGN_ON.init();
		LGN_ON.allocate();

		LGN_ON.addProjection(IN.Xptr, IN.hg, afferRadius, +LGNstrength, 0.0,
				LGNCenterSigma, false);
		LGN_ON.addProjection(IN.Xptr, IN.hg, afferRadius, -LGNstrength, 0.0,
				LGNSurroundSigma, false);
		LGN_ON.addProjection(LGN_ON.Xptr, LGN_ON.hg, afferRadius, -LGNstrength,
				0.0, LGNSurroundSigma, false);
		for (unsigned int i = 0; i < LGN_ON.Projections.size(); i++) {
			LGN_ON.Projections[i].renormalize();
		}

		LGN_OFF.strength = LGNstrength;
		LGN_OFF.svgpath =
				root.get("IN_svgpath", "boundaries/trialmod.svg").asString();
		LGN_OFF.init();
		LGN_OFF.allocate();

		LGN_OFF.addProjection(IN.Xptr, IN.hg, afferRadius, -LGNstrength, 0.0,
				LGNCenterSigma, false);
		LGN_OFF.addProjection(IN.Xptr, IN.hg, afferRadius, +LGNstrength, 0.0,
				LGNSurroundSigma, false);
		LGN_OFF.addProjection(LGN_OFF.Xptr, LGN_OFF.hg, afferRadius,
				-LGNstrength, 0.0, LGNSurroundSigma, false);
		for (unsigned int i = 0; i < LGN_OFF.Projections.size(); i++) {
			LGN_OFF.Projections[i].renormalize();
		}

		// CORTEX SHEET
		CX.beta = beta;
		CX.lambda = lambda;
		CX.mu = mu;
		CX.thetaInit = thetaInit;
		CX.svgpath =
				root.get("CX_svgpath", "boundaries/trialmod.svg").asString();
		CX.init();
		CX.allocate();

		CX.addProjection(LGN_ON.Xptr, LGN_ON.hg, afferRadius,
				afferStrength * 0.5, afferAlpha, afferSigma, true);
		CX.addProjection(LGN_OFF.Xptr, LGN_OFF.hg, afferRadius,
				afferStrength * 0.5, afferAlpha, afferSigma, true);
		CX.addProjection(CX.Xptr, CX.hg, excitRadius, excitStrength, excitAlpha,
				excitSigma, true);
		CX.addProjection(CX.Xptr, CX.hg, inhibRadius, inhibStrength, inhibAlpha,
				inhibSigma, true);

		// SETUP FIELDS FOR JOINT NORMALIZATION
		vector<int> p1(2, 0);
		p1[1] = 1;
		CX.setNormalize(p1);
		CX.setNormalize(vector<int>(1, 2));
		CX.setNormalize(vector<int>(1, 3));
		CX.renormalize();

		orientationPref.resize(CX.nhex, 0.);
		orientationSel.resize(CX.nhex, 0.);

		directionPref.resize(CX.nhex, 0.);
		directionSel.resize(CX.nhex, 0.);

	}

	void stepAfferent(unsigned type) {
		switch (type) {
		case (0): { // Gaussians
			IN.Gaussian((morph::Tools::randDouble() - 0.5) * xRange,
					(morph::Tools::randDouble() - 0.5) * yRange,
					morph::Tools::randDouble() * M_PI, sigmaA, sigmaB);
		}
			break;

		case (1): { // Preloaded
			HCM.stepPreloaded();
			IN.X = HCM.X;
		}
			break;

		case (2): { // Camera input
			HCM.stepCamera();
			IN.X = HCM.X;
		}
			break;

		case (3): { // Video input
			HCM.stepVideo();
			IN.X = HCM.X;
		}
			break;

		default: {
			for (int i = 0; i < HCM.C.n; i++) {
				HCM.C.vsquare[i].X = morph::Tools::randDouble();
			}
			HCM.step();
			IN.X = HCM.X;
		}
		}
		LGN_ON.step();
		LGN_OFF.step();
	}

	void plotAfferent(morph::Gdisplay disp1, morph::Gdisplay disp2) {
		vector<double> fx(3, 0.);
		RD_Plot<double> plt(fx, fx, fx);
		plt.scalarfields(disp1, IN.hg, IN.X, 0., 1.0);
		vector<vector<double> > L;
		L.push_back(LGN_ON.X);
		L.push_back(LGN_OFF.X);
		plt.scalarfields(disp2, LGN_ON.hg, L);
        stringstream ss;
        ss << "Input Activity. Step: " << time;
        std::string s = ss.str();
        const char* cc = s.c_str();
        char* c = (char*)cc;
//        char * c = s;
//        //std::string numStr = std::to_string(time);
//        std::array<char, 10> str;
//        std::to_chars(str.data(), str.data() + str.size(), time);
        disp1.setTitle(c);
	}

	void plotCortex(morph::Gdisplay disp) {
		vector<double> fx(3, 0.);
		RD_Plot<double> plt(fx, fx, fx);
		plt.scalarfields(disp, CX.hg, CX.X);
	}

	void stepCortex(bool zeroCortex, bool decayCortex) {
        if(zeroCortex){
        	CX.zero_X();
        }
        else if(decayCortex){
        	CX.decay();
        }
		for (unsigned int j = 0; j < settle; j++) {
			CX.step(zeroCortex, decayCortex);
		}
		for (unsigned int p = 0; p < CX.Projections.size(); p++) {
			CX.Projections[p].learn();
		}
		CX.renormalize();
		if (homeostasis) {
			CX.homeostasis();
		}
		time++;
	}

	void stepCortex(morph::Gdisplay disp, bool zeroCortex, bool decayCortex) {
		vector<double> fx(3, 0.);
		RD_Plot<double> plt(fx, fx, fx);
        if(zeroCortex){
        	CX.zero_X();
        }
        else if(decayCortex){
        	CX.decay();
        }
		for (unsigned int j = 0; j < settle; j++) {
			CX.step(zeroCortex, decayCortex);
			plt.scalarfields(disp, CX.hg, CX.X);
		}
		for (unsigned int p = 0; p < CX.Projections.size(); p++) {
			CX.Projections[p].learn();
		}
		CX.renormalize();
		if (homeostasis) {
			CX.homeostasis();
		}
		time++;
	}

	void plotWeights(morph::Gdisplay disp, int id) {
		vector<double> fix(3, 0.);
		RD_Plot<double> plt(fix, fix, fix);
		vector<vector<double> > W;
		W.push_back(CX.Projections[0].getWeightPlot(id));
		W.push_back(CX.Projections[1].getWeightPlot(id));
		W.push_back(CX.Projections[3].getWeightPlot(id));
		plt.scalarfields(disp, CX.hg, W);
	}

	void plotMap(morph::Gdisplay disp) {

		disp.resetDisplay(vector<double>(3, 0.), vector<double>(3, 0.),
				vector<double>(3, 0.));
		double maxSel = -1e9;
		for (int i = 0; i < CX.nhex; i++) {
			if (orientationSel[i] > maxSel) {
				maxSel = orientationSel[i];
			}
		}
		maxSel = 1. / maxSel;
		double overPi = 1. / M_PI;

		int i = 0;
		for (auto h : CX.hg->hexen) {
			array<float, 3> cl = morph::Tools::HSVtoRGB(
					orientationPref[i] * overPi, 1.0,
					orientationSel[i] * maxSel);
			disp.drawHex(h.position(), array<float, 3> { 0., 0., 0. },
					(h.d * 0.5f), cl);
			i++;
		}
		disp.redrawDisplay();
	}

	void map(void) {
		//int nOr=20;											// number of orientations
		//int nPhase=8;										// number of phases
		double phaseInc = M_PI / (double) numGratingPhases;	// phase increment
		vector<int> maxIndOr(CX.nhex, 0);	// orientation index i (1<i<=nOr)
		vector<double> maxValOr(CX.nhex, -1e9);	// store max activation for orientations
		vector<double> maxPhase(CX.nhex, 0.);// store max activation for given phase
		vector<double> Vx(CX.nhex);							//
		vector<double> Vy(CX.nhex);							//
		vector<int> aff(2, 0);
		aff[1] = 1;
		for (unsigned int i = 0; i < numGratingOrientations; i++) {	// loop over each orientation
			double theta = i * M_PI / (double) numGratingOrientations;// calculate angle
			std::fill(maxPhase.begin(), maxPhase.end(), -1e9);
			for (unsigned int j = 0; j < numGratingPhases; j++) {// loop over each phase
				double phase = j * phaseInc;				// calculate phase
				IN.Grating(theta, phase, 30.0, 1.0);
				LGN_ON.step();
				LGN_OFF.step();
				CX.zero_X();
				CX.step(aff, true, false);
				CX.homeostasis();
				for (int k = 0; k < maxPhase.size(); k++) {
					if (maxPhase[k] < CX.X[k]) {
						maxPhase[k] = CX.X[k];
					}// store activation value for this phase, if greater than current stored
				}
			}

			for (int k = 0; k < maxPhase.size(); k++) {
				Vx[k] += maxPhase[k] * cos(2.0 * theta);
				Vy[k] += maxPhase[k] * sin(2.0 * theta);
			}

			for (int k = 0; k < maxPhase.size(); k++) {
				if (maxValOr[k] < maxPhase[k]) {
					maxValOr[k] = maxPhase[k];
					maxIndOr[k] = i;
				}
			}
		}

		for (int i = 0; i < maxValOr.size(); i++) {
			orientationPref[i] = 0.5 * (atan2(Vy[i], Vx[i]) + M_PI);// the angles of selectivity
			orientationSel[i] = sqrt(Vy[i] * Vy[i] + Vx[i] * Vx[i]);// the strengths of the selectivity
		}
	}

	void map(morph::Gdisplay affDisp1, morph::Gdisplay affDisp2,
			morph::Gdisplay cortexDisp) {
		//int nOr=20;											// number of orientations
		//int nPhase=8;										// number of phases
		double phaseInc = M_PI / (double) numGratingPhases;	// phase increment
		vector<int> maxIndOr(CX.nhex, 0);	// orientation index i (1<i<=nOr)
		vector<double> maxValOr(CX.nhex, -1e9);	// store max activation for orientations
		vector<double> maxPhase(CX.nhex, 0.);// store max activation for given phase
		vector<double> Vx(CX.nhex);							//
		vector<double> Vy(CX.nhex);							//
		vector<int> aff(2, 0);
		aff[1] = 1;
		for (unsigned int i = 0; i < numGratingOrientations; i++) {	// loop over each orientation
			double theta = i * M_PI / (double) numGratingOrientations;// calculate angle
			std::fill(maxPhase.begin(), maxPhase.end(), -1e9);
			for (unsigned int j = 0; j < numGratingPhases; j++) {// loop over each phase
				double phase = j * phaseInc;				// calculate phase
				IN.Grating(theta, phase, 30.0, 1.0);
				LGN_ON.step();
				LGN_OFF.step();
				this->plotAfferent(affDisp1, affDisp2);
				CX.zero_X();
				CX.step(aff, true, false);
				CX.homeostasis();
				plotCortex(cortexDisp);
				for (int k = 0; k < maxPhase.size(); k++) {
					if (maxPhase[k] < CX.X[k]) {
						maxPhase[k] = CX.X[k];
					}// store activation value for this phase, if greater than current stored
				}
			}

			for (int k = 0; k < maxPhase.size(); k++) {
				Vx[k] += maxPhase[k] * cos(2.0 * theta);
				Vy[k] += maxPhase[k] * sin(2.0 * theta);
			}

			for (int k = 0; k < maxPhase.size(); k++) {
				if (maxValOr[k] < maxPhase[k]) {
					maxValOr[k] = maxPhase[k];
					maxIndOr[k] = i;
				}
			}
		}

		for (int i = 0; i < maxValOr.size(); i++) {
			orientationPref[i] = 0.5 * (atan2(Vy[i], Vx[i]) + M_PI);// the angles of selectivity
			orientationSel[i] = sqrt(Vy[i] * Vy[i] + Vx[i] * Vx[i]);// the strengths of the selectivity
		}
	}

	void plotDirectionMap(morph::Gdisplay disp) {
		disp.resetDisplay(vector<double>(3, 0.), vector<double>(3, 0.),
				vector<double>(3, 0.));
		double maxSel = -1e9;
		for (int i = 0; i < CX.nhex; i++) {
			if (directionSel[i] > maxSel) {
				maxSel = directionSel[i];
			}
		}
		maxSel = 1. / maxSel;
		double overPi = 1. / M_PI;

		int i = 0;
		for (auto h : CX.hg->hexen) {
			array<float, 3> cl = morph::Tools::HSVtoRGB(
					directionPref[i] * overPi, 1.0, directionSel[i] * maxSel);
			disp.drawHex(h.position(), array<float, 3> { 0., 0., 0. },
					(h.d * 0.5f), cl);
			i++;
		} //TODO draw arrow. See morph draw options
		disp.redrawDisplay();
	}

	void directionMap(morph::Gdisplay affDisp1, morph::Gdisplay affDisp2,
			morph::Gdisplay cortexDisp) {
		//int nOr=40;											// number of orientations
		//int nPhase=100;										// number of phases
		double phaseInc = 2 * M_PI / (double) numGratingPhases;	// phase increment
		vector<int> maxIndOr(CX.nhex, 0);	// orientation index i (1<i<=nOr)
		vector<double> maxValOr(CX.nhex, -1e9);	// store max activation for orientations
		vector<double> maxPhase(CX.nhex, 0.);// store max activation for given phase
		vector<double> Vx(CX.nhex);							//
		vector<double> Vy(CX.nhex);							//
		vector<int> aff(2, 0);
		aff[1] = 1;
		for (unsigned int i = 0; i < numDirectionOrientations; i++) {// loop over each orientation
			double theta = i * 2 * M_PI / (double) numDirectionOrientations;// calculate angle
			std::fill(maxPhase.begin(), maxPhase.end(), -1e9);
			for (unsigned int j = 0; j < numGratingPhases; j++) {// loop over each phase
				double phase = j * phaseInc;				// calculate phase
				IN.Grating(theta, phase, gratingWidth, 1.0);
				LGN_ON.step();
				LGN_OFF.step();
				this->plotAfferent(affDisp1, affDisp2);
				//CX.decay();
				CX.step(aff, false, false);
				CX.homeostasis();
				plotCortex(cortexDisp);
				for (int k = 0; k < maxPhase.size(); k++) {
					if (maxPhase[k] < CX.X[k]) {
						maxPhase[k] = CX.X[k];
					}// store activation value for this phase, if greater than current stored
				}
			}

			for (int k = 0; k < maxPhase.size(); k++) {
				Vx[k] += maxPhase[k] * cos(2.0 * theta);
				Vy[k] += maxPhase[k] * sin(2.0 * theta);
			}

			for (int k = 0; k < maxPhase.size(); k++) {
				if (maxValOr[k] < maxPhase[k]) {
					maxValOr[k] = maxPhase[k];
					maxIndOr[k] = i;
				}
			}
			CX.zero_X();
		}

		for (int i = 0; i < maxValOr.size(); i++) {
			directionPref[i] = 0.5 * (atan2(Vy[i], Vx[i]) + M_PI);// the angles of selectivity
			directionSel[i] = sqrt(Vy[i] * Vy[i] + Vx[i] * Vx[i]);// the strengths of the selectivity
		}
	}

	void directionMap(void) {
		//int nOr=40;											// number of orientations
		//int nPhase=100;										// number of phases
		double phaseInc = 2 * M_PI / (double) numGratingPhases;	// phase increment
		vector<int> maxIndOr(CX.nhex, 0);	// orientation index i (1<i<=nOr)
		vector<double> maxValOr(CX.nhex, -1e9);	// store max activation for orientations
		vector<double> maxPhase(CX.nhex, 0.);// store max activation for given phase
		vector<double> Vx(CX.nhex);							//
		vector<double> Vy(CX.nhex);							//
		vector<int> aff(2, 0);
		aff[1] = 1;
		for (unsigned int i = 0; i < numDirectionOrientations; i++) {// loop over each orientation
			double theta = i * 2 * M_PI / (double) numDirectionOrientations;// calculate angle
			std::fill(maxPhase.begin(), maxPhase.end(), -1e9);
			for (unsigned int j = 0; j < numGratingPhases; j++) {// loop over each phase
				double phase = j * phaseInc;				// calculate phase
				IN.Grating(theta, phase, gratingWidth, 1.0);
				LGN_ON.step();
				LGN_OFF.step();
				//CX.decay();
				CX.step(aff, false, false);
				CX.homeostasis();
				for (int k = 0; k < maxPhase.size(); k++) {
					if (maxPhase[k] < CX.X[k]) {
						maxPhase[k] = CX.X[k];
					}// store activation value for this phase, if greater than current stored
				}
			}

			for (int k = 0; k < maxPhase.size(); k++) {
				Vx[k] += maxPhase[k] * cos(2.0 * theta);
				Vy[k] += maxPhase[k] * sin(2.0 * theta);
			}

			for (int k = 0; k < maxPhase.size(); k++) {
				if (maxValOr[k] < maxPhase[k]) {
					maxValOr[k] = maxPhase[k];
					maxIndOr[k] = i;
				}
			}
			CX.zero_X();
		}

		for (int i = 0; i < maxValOr.size(); i++) {
			directionPref[i] = 0.5 * (atan2(Vy[i], Vx[i]) + M_PI);// the angles of selectivity
			directionSel[i] = sqrt(Vy[i] * Vy[i] + Vx[i] * Vx[i]);// the strengths of the selectivity
		}
	}

	void save(string filename) {
		stringstream fname;
		fname << filename;
		HdfData data(fname.str());
		vector<int> timetmp(1, time);
		data.add_contained_vals("time", timetmp);
		for (unsigned int p = 0; p < CX.Projections.size(); p++) {
			vector<double> proj = CX.Projections[p].getWeights();
			stringstream ss;
			ss << "proj_" << p;
			data.add_contained_vals(ss.str().c_str(), proj);
		}
	}

	void load(string filename) {
		stringstream fname;
		fname << filename;
		HdfData data(fname.str(), 1);
		vector<int> timetmp;
		data.read_contained_vals("time", timetmp);
		time = timetmp[0];
		for (unsigned int p = 0; p < CX.Projections.size(); p++) {
			vector<double> proj;
			stringstream ss;
			ss << "proj_" << p;
			data.read_contained_vals(ss.str().c_str(), proj);
			CX.Projections[p].setWeights(proj);
		}
		cout << "Loaded weights and modified time to " << time << endl;
	}

	~gcal(void) {
	}

};

/*** MAIN PROGRAM ***/

int main(int argc, char **argv) {

	if (argc < 5) {
		cerr
				<< "\nUsage: ./test configfile seed mode intype weightfile(optional)\n\n";
		return -1;
	}

	string paramsfile(argv[1]);
	srand(stoi(argv[2]));       // set seed
	int MODE = stoi(argv[3]);
	int INTYPE = stoi(argv[4]); // 0,1,2 Gaussian,Loaded,Camera input

	//  Set up JSON code for reading the parameters
	ifstream jsonfile_test;
	int srtn = system("pwd");
	if (srtn) {
		cerr << "system call returned " << srtn << endl;
	}

	jsonfile_test.open(paramsfile, ios::in);
	if (jsonfile_test.is_open()) {
		jsonfile_test.close(); // Good, file exists.
	} else {
		cerr << "json config file " << paramsfile << " not found." << endl;
		return 1;
	}

	// Parse the JSON
	ifstream jsonfile(paramsfile, ifstream::binary);
	Json::Value root;
	string errs;
	Json::CharReaderBuilder rbuilder;
	rbuilder["collectComments"] = false;
	bool parsingSuccessful = Json::parseFromStream(rbuilder, jsonfile, &root,
			&errs);
	if (!parsingSuccessful) {
		cerr << "Failed to parse JSON: " << errs;
		return 1;
	}

	unsigned int nBlocks = root.get("blocks", 100).asUInt();
	unsigned int steps = root.get("steps", 100).asUInt();

	// Creates the network
	gcal Net;
	Net.init(root);

	// Input specific setup
	switch (INTYPE) {
	case (0): { // Gaussian patterns
	}
		break;
	case (1): {   // preload patterns
		int ncols = root.get("cameraCols", 100).asUInt();
		int nrows = root.get("cameraRows", 100).asUInt();
		Net.HCM.initProjection(ncols, nrows, 0.01, 20.);
		string filename =
				root.get("patterns", "configs/testPatterns.h5").asString();
		Net.HCM.preloadPatterns(filename);
	}
		break;

	case (2): {
		int ncols = root.get("cameraCols", 100).asUInt();
		int nrows = root.get("cameraRows", 100).asUInt();
		int stepsize = root.get("cameraSampleStep", 7).asUInt(); // Value should be based on camera resolution (height)
		int xoff = root.get("cameraOffsetX", 100).asUInt();
		int yoff = root.get("cameraOffsetY", 0).asUInt();
		Net.HCM.initProjection(ncols, nrows, 0.01, 20.);
		if (!Net.HCM.initCamera(xoff, yoff, stepsize)) {
			return 0;
		}
	}
		break;

	case (3): {
		int ncols = root.get("videoCols", 100).asUInt();
		int nrows = root.get("videoRows", 100).asUInt();
		int xOffset = root.get("videoOffsetX", 0).asUInt();
		int yOffset = root.get("videoOffsetY", 0).asUInt();
		Net.HCM.initProjection(ncols, nrows, 0.01, 20.);
		string videoSource = root.get("videoSource",
				"configs/videos/planet_earth_700.avi").asString();
		if (!Net.HCM.initVideo(xOffset, yOffset, ncols, videoSource)) {
			cout << "Failed to open video file" << endl;
			return 0;
		}
	}
		break;
	}

	bool plotGratingStage = root.get("plotGratingStage", false).asBool();
	bool plotInitialMaps = root.get("plotInitialMaps", true).asBool();

	if (argc > 5) {
		cout << "Using weight file: " << argv[5] << endl;
		Net.load(argv[5]);
	} else {
		cout << "Using random weights" << endl;
	}

	switch (MODE) {

	case (0): { // No plotting
		for (int b = 0; b < nBlocks; b++) {
			Net.map();
			for (unsigned int i = 0; i < steps; i++) {
				Net.stepAfferent(INTYPE);
				Net.stepCortex(Net.zeroCortex, Net.decayCortex);
			}
			stringstream ss;
			ss << "weights_" << Net.time << ".h5";
			Net.save(ss.str());
		}
	}
		break;

	case (1): { // Plotting
		vector<morph::Gdisplay> displays;
		displays.push_back(
				morph::Gdisplay(600, 600, 0, 0, "Input Activity", 1.7, 0.0,
						0.0));
		displays.push_back(
				morph::Gdisplay(600, 600, 0, 0, "Cortical Activity", 1.7, 0.0,
						0.0));
		displays.push_back(
				morph::Gdisplay(1200, 400, 0, 0, "Cortical Projection", 1.7,
						0.0, 0.0));
		displays.push_back(
				morph::Gdisplay(600, 300, 0, 0, "LGN ON/OFF", 1.7, 0.0, 0.0));
		displays.push_back(
				morph::Gdisplay(600, 600, 0, 0, "Orientation Selectivity Map",
						1.7, 0.0, 0.0));
		displays.push_back(
				morph::Gdisplay(600, 600, 0, 0, "Direction Selectivity Map",
						1.7, 0.0, 0.0));
		for (unsigned int i = 0; i < displays.size(); i++) {
			displays[i].resetDisplay(vector<double>(3, 0), vector<double>(3, 0),
					vector<double>(3, 0));
			displays[i].redrawDisplay();
		}

		//Create folder and save config file
		time_t now = time(0);
		struct tm tstruct;
		char buf[80];
		tstruct = *localtime(&now);
		// Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
		// for more information about date/time format
		strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

		string directoryName = "results/" + std::string(buf) + "/";
		string dirCommand = "mkdir " + directoryName;
		system(dirCommand.c_str());

		string configFile = directoryName + "configs.json";
		ofstream myfile;
		myfile.open(configFile);
		myfile << root;
		myfile.close();

		for (int b = 0; b < nBlocks; b++) {
			if (plotGratingStage && plotInitialMaps) {
				Net.map(displays[0], displays[3], displays[1]);
				Net.directionMap(displays[0], displays[3], displays[1]);
				Net.plotMap(displays[4]);
				Net.plotDirectionMap(displays[5]);
			} else if (plotInitialMaps) {
				Net.map();
				Net.directionMap();
				Net.plotMap(displays[4]);
				Net.plotDirectionMap(displays[5]);
			}

			for (unsigned int i = 0; i < steps; i++) {
				Net.stepAfferent(INTYPE);
				Net.plotAfferent(displays[0], displays[3]);
				Net.stepCortex(displays[1], Net.zeroCortex, Net.decayCortex);
				Net.plotWeights(displays[2], 500);
			}

			stringstream ss;
			ss << directoryName << "weights_" << Net.time << ".h5";
			Net.save(ss.str());

			string orientationFilename = directoryName + "orientationmap_"
					+ to_string(Net.time) + ".png";
			displays[4].saveImage(orientationFilename);
			string dirFilename = directoryName + "directionmap_"
					+ to_string(Net.time) + ".png";
			displays[5].saveImage(dirFilename);
		}
		for (unsigned int i = 0; i < displays.size(); i++) {
			displays[i].closeDisplay();
		}
	}
		break;

	case (2): { // Map only
		vector<morph::Gdisplay> displays;
		displays.push_back(
				morph::Gdisplay(600, 600, 0, 0, "Input Activity", 1.7, 0.0,
						0.0));
		displays.push_back(
				morph::Gdisplay(600, 600, 0, 0, "Cortical Activity", 1.7, 0.0,
						0.0));
		displays.push_back(
				morph::Gdisplay(1200, 400, 0, 0, "Cortical Projection", 1.7,
						0.0, 0.0));
		displays.push_back(
				morph::Gdisplay(600, 300, 0, 0, "LGN ON/OFF", 1.7, 0.0, 0.0));
		displays.push_back(
				morph::Gdisplay(600, 600, 0, 0, "Map", 1.7, 0.0, 0.0));
		for (unsigned int i = 0; i < displays.size(); i++) {
			displays[i].resetDisplay(vector<double>(3, 0), vector<double>(3, 0),
					vector<double>(3, 0));
			displays[i].redrawDisplay();
		}
		Net.map();
		Net.plotMap(displays[4]);
		Net.stepAfferent(INTYPE);
		Net.plotAfferent(displays[0], displays[3]);
		Net.stepCortex(displays[1], Net.zeroCortex, Net.decayCortex);
		Net.plotWeights(displays[2], 500);
		for (unsigned int i = 0; i < displays.size(); i++) {
			displays[i].redrawDisplay();
			stringstream ss;
			ss << "plot_" << Net.time << "_" << i << ".png";
			displays[i].saveImage(ss.str());
		}
		for (unsigned int i = 0; i < displays.size(); i++) {
			displays[i].closeDisplay();
		}
	}
		break;
	}

	return 0.;
}
