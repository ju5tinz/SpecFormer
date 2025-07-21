#include <fstream>
#include <cstdio>
#include "CFeatureList.h"

CFeatureList::CFeatureList()
{
	//construct a dictionary of features
	using namespace std;
	int MaxNumFeatures = 65536;	//targeting ~10000
	m_Features = new string[MaxNumFeatures];
	int i, j, j2, k, NumFeatures = 0;
	char cTemp[20];

	m_MaxNumResidues = 20;
	m_MaxCharge = 4;

	//molecular ions
	m_Features[NumFeatures++] = "Mol";
	m_Features[NumFeatures++] = "Mol-17";
	m_Features[NumFeatures++] = "Mol-18";
	for (k = 30; k <= 64; k++)
	{
		sprintf(cTemp, "Mol-%d", k);
		m_Features[NumFeatures++] = cTemp;
	}
	m_Features[NumFeatures++] = "Mol-91";
	m_Features[NumFeatures++] = "Mol-92";

	//b and y
	for (i = 1; i < m_MaxNumResidues; i++)
	{
		for (j = 1; j <= m_MaxCharge; j++)
		{
			if (j > i / 27 && (j == 1 || j <= i / 2))	//make sure charge is not too small or large
			{
				sprintf(cTemp, "b%d[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;

				sprintf(cTemp, "b%d+18[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;

				sprintf(cTemp, "b%d+17[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;

				sprintf(cTemp, "b%d-17[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;

				sprintf(cTemp, "b%d-18[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;

				sprintf(cTemp, "b%d-28[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;

				sprintf(cTemp, "y%d[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;

				sprintf(cTemp, "y%d+28[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;

				sprintf(cTemp, "y%d-17[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;

				sprintf(cTemp, "y%d-18[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;

				for (k = 30; k <= 64; k++)
				{
					sprintf(cTemp, "b%d-%d[%d+]", i, k, j);
					m_Features[NumFeatures++] = cTemp;
					sprintf(cTemp, "y%d-%d[%d+]", i, k, j);
					m_Features[NumFeatures++] = cTemp;
				}

				//-91 and 92 for U and J
				sprintf(cTemp, "b%d-91[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;
				sprintf(cTemp, "y%d-91[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;
				sprintf(cTemp, "b%d-92[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;
				sprintf(cTemp, "y%d-92[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;
			}
		}
	}

	//internal fragments
	for (i = 2; i <= m_MaxNumResidues - 2; i++)
	{
		for (j2 = i + 1; j2 <= m_MaxNumResidues - 1; j2++)
		{
			for (j = 1; j <= m_MaxCharge; j++)
			{
				if (j > (j2 - i + 1) / 21 && (j == 1 || j <= (j2 - i + 1) * 2 / 5))
				{
					sprintf(cTemp, "[%d_%d][%d+]", i, j2, j);
					m_Features[NumFeatures++] = cTemp;
					sprintf(cTemp, "[%d_%d]-17[%d+]", i, j2, j);
					m_Features[NumFeatures++] = cTemp;
					sprintf(cTemp, "[%d_%d]-18[%d+]", i, j2, j);
					m_Features[NumFeatures++] = cTemp;
					sprintf(cTemp, "[%d_%d]-28[%d+]", i, j2, j);
					m_Features[NumFeatures++] = cTemp;
				}
			}
		}
	}

	//single amino acid
	char aa[27] = "ACDEFGHIJKLMNOPQRSTUVWYsty";
	for (i = 0; i < 23; i++)	//do not include sty
	{
		if (aa[i] != 'I')	//do not include Ile
		{
			sprintf(cTemp, "%c'[1+]", aa[i]);	//c' is the acyl form
			m_Features[NumFeatures++] = cTemp;
			sprintf(cTemp, "%c[1+]", aa[i]);
			m_Features[NumFeatures++] = cTemp;
		}
	}
	m_NumFeatures = NumFeatures;
}

CFeatureList::~CFeatureList()
{
	delete[] m_Features;
}

void CFeatureList::SaveAs(char filename[])
{
	using namespace std;
	ofstream file(filename, ios::out);

	//sequence
	file << "Max peptide length=" << m_MaxNumResidues << '\n';
	file << "Max fragment charge=" << m_MaxCharge << '\n';
	file << "Dictionary size=" << m_NumFeatures << '\n';
	int i;
	for (i = 0; i < m_NumFeatures; i++)
		file << i << ',' << m_Features[i] << '\n';
	file.close();
	return;
}