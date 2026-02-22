#pragma once
#include <string.h>
#include <iostream>
#include <fstream>
#include "CFeatureList.h"

class CMS2Spectrum
{
public:
	CMS2Spectrum(long, float*, float*, std::string, int, float, float, CFeatureList *, float *, float, float);
	~CMS2Spectrum();
	void SetFeatures(std::string, int, float, float, CFeatureList*, float*, float, float);
	void Annotate();
	void ExportAnnotations(char []);
	void GenerateSpectrum();
	void ExportSpectrum(char[]);

	long m_points;
	float* m_mass, * m_intensity;
	float m_MaxSignal;

	int m_model;
	int m_FragmentMethod;
	int m_charge;
	float m_CollisionEnergy;
	float m_Resolution;
	float m_IsolationWidth;
	float m_ReactionTime;
	float m_ActivationQ;
	float m_PrecursorMz;
	float m_StartMass, m_EndMass;
	double m_Mass0;
	std::string m_sequence;

private:
	int m_Is4Annotation;	//1 for annotation, 0 for conversion from annotation to spectrum
	float* m_labelscore;
	float* m_LabelIntensity;
	float* x;
	float* y;
	std::string *m_label;

	CFeatureList* m_FeatureList;
	float* m_FeatureIntensity;

	int m_NumFragIons;
	float * m_FragIonMz;
	int *m_FragIonCharge;
	float *m_FragIonIntensity;
	int* m_SortIndex;

	double GetResidueMasses(int, std::string, double *);
	int CheckFragment(char [], double, float, int, float [], float);
	float GetCleavageLikelihood(int);
	int IsotopeSimulationFromMass(double, int, float [], float, int *);
	int AddToSpectrum(int, float[], float []);
	void GoldenSearch(int*, int*, float);
};

