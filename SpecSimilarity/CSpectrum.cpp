#include "CSpectrum.h"
#include <math.h>

CSpectrum::CSpectrum(long n, float* mass, float* intensity)
{
	m_points = n;
	m_mass = mass;
	m_intensity = intensity;
}

CSpectrum::~CSpectrum()
{

}

//calculate similarity with 0.05 tolerance
float CSpectrum::Similarity(int n, float* mass, float* intensity, float tol)
{
	int i;

	double sum1 = 0.0;
	double sum2 = 0.0;
	for (i = 0; i < m_points; i++)
		sum1 += (double)m_intensity[i];
	for (i = 0; i < n; i++)
		sum2 += (double)intensity[i];

	double rootproduct = 0.0;
	int p1=0, p2=0;

	rootproduct = 0.0;
	while (p1 < m_points && p2 < n)
	{
		if (mass[p2] - m_mass[p1] > tol)
			p1++;
		else if (m_mass[p1] - mass[p2] > tol)
			p2++;
		else if (m_mass[p1] >= mass[p2])
		{
			while (p2 < n - 1 && mass[p2 + 1] - m_mass[p1] < m_mass[p1] - mass[p2])	//find the p2 closest to p1
				p2++;
			rootproduct += sqrt((double)m_intensity[p1] * (double)intensity[p2]);
			p1++;
			p2++;
		}
		else
		{
			while (p1 < m_points - 1 && m_mass[p1 + 1] - mass[p2] < mass[p2] - m_mass[p1])	//find the p1 closest to p2
				p1++;
			rootproduct += sqrt((double)m_intensity[p1] * (double)intensity[p2]);
			p1++;
			p2++;
		}
	}
	return (float)(rootproduct/sqrt(sum1*sum2));
}