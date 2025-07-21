#pragma once
#include <string.h>
#include <iostream>
#include <fstream>
#include "CFeatureList.h"

class CMS2Spectrum
{
public:
	CMS2Spectrum(long, float*, float*, std::string, int, float, float);
	~CMS2Spectrum();
	void Annotate();
	void ExportAnnotations(char [], CFeatureList *);

private:
	long m_points;
	float* m_mass, * m_intensity;
	float* m_labelscore;
	float* m_LabelIntensity;
	std::string *m_label;

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


	double GetResidueMasses(int, std::string, double *);
	int CheckMass(char [], double, float, int, float);
	float CleavageLikelihood(int);
	int IsotopeSimulationFromMass(double, int, float [], float, int *);
};

