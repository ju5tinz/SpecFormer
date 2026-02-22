#pragma once
class CSpectrum
{
public:
	CSpectrum(long, float*, float*);
	~CSpectrum();
	float Similarity(int, float*, float*, float);

	long m_points;
	float* m_mass, * m_intensity;
};

