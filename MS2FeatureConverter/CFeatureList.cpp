#include <fstream>
#include "CFeatureList.h"

//mass offset applied (global variable used for both feature list and spectrum annotation
int bOffset[NUM_B_OFFSETS] = {34,35,36,42,44,45,46,52,53,54,55,57,59,61,62,63,64};
int yOffset[NUM_Y_OFFSETS] = { 34,35,36,42,44,52,53,54,59,60,61,64 };


CFeatureList::CFeatureList()
{
	//construct a dictionary of features (annotations)
	using namespace std;
	int MaxNumFeatures = 65536;
	m_Features = new string[MaxNumFeatures];
	int i, j, j2, k, NumFeatures = 0;
	char cTemp[20];

	m_MaxNumResidues = MAXLENGTH;
	m_MaxCharge = MAXCHARGE;

	//molecular ions
	m_Features[NumFeatures++] = "Mol";
	m_Features[NumFeatures++] = "Mol-17";
	m_Features[NumFeatures++] = "Mol-18";

	for (k = 0; k < NUM_Y_OFFSETS; k++)
	{
		sprintf(cTemp, "Mol-%d", yOffset[k]);
		m_Features[NumFeatures++] = cTemp;
	}

	m_Features[NumFeatures++] = "Mol-Cterm";

	//b and y
	for (i = 1; i < m_MaxNumResidues; i++)
	{
		for (j = 1; j <= m_MaxCharge; j++)
		{
			if (j > i / 27 && (j == 1 || j <= i / 2))	//make sure charge is not too small or large
			{
				//b ions
				sprintf(cTemp, "b%d[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;

				sprintf(cTemp, "b%d+17[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;

				sprintf(cTemp, "b%d-17[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;

				sprintf(cTemp, "b%d-18[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;

				sprintf(cTemp, "b%d-28[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;

				for (k = 0; k < NUM_B_OFFSETS; k++)
				{
					sprintf(cTemp, "b%d-%d[%d+]", i, bOffset[k], j);
					m_Features[NumFeatures++] = cTemp;
				}

				//-81 and 82 for NH3 and H2O loss from O-64
				sprintf(cTemp, "b%d-81[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;
				sprintf(cTemp, "b%d-82[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;

				//-91 and 92 for U and J
				sprintf(cTemp, "b%d-91[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;
				sprintf(cTemp, "b%d-92[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;


				//y ions
				sprintf(cTemp, "y%d[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;

				sprintf(cTemp, "y%d+28[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;

				sprintf(cTemp, "y%d-17[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;

				sprintf(cTemp, "y%d-18[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;

				for (k = 0; k < NUM_Y_OFFSETS; k++)
				{
					sprintf(cTemp, "y%d-%d[%d+]", i, yOffset[k], j);
					m_Features[NumFeatures++] = cTemp;
				}

				//-81 and 82 for NH3 and H2O loss from O-64
				sprintf(cTemp, "y%d-81[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;
				sprintf(cTemp, "y%d-82[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;

				//-91 and 92 for U and J
				sprintf(cTemp, "y%d-91[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;
				sprintf(cTemp, "y%d-92[%d+]", i, j);
				m_Features[NumFeatures++] = cTemp;

				//y - C-term residue (C-term rearrangement from an y ion)
				if (i > 1)
				{
					sprintf(cTemp, "y%d-Cterm[%d+]", i, j);
					m_Features[NumFeatures++] = cTemp;
				}
			}
		}
	}

	//special b2 neutral losses
	m_Features[NumFeatures++] = "b2-73[1+]";
	m_Features[NumFeatures++] = "b2-76[1+]";
	m_Features[NumFeatures++] = "b2-143[1+]";
	m_Features[NumFeatures++] = "b2-159[1+]";

	//internal fragments
	for (i = 2; i <= m_MaxNumResidues - 2; i++)
	{
		for (j2 = i + 1; j2 <= m_MaxNumResidues - 1; j2++)
		{
			for (j = 1; j <= m_MaxCharge; j++)
			{
				if (j > (j2 - i + 1) / 27 && (j == 1 || j <= (j2 - i + 1) * 2 / 5))
				{
					sprintf(cTemp, "[%d_%d][%d+]", i, j2, j);
					m_Features[NumFeatures++] = cTemp;
					sprintf(cTemp, "[%d_%d]-17[%d+]", i, j2, j);
					m_Features[NumFeatures++] = cTemp;
					sprintf(cTemp, "[%d_%d]-18[%d+]", i, j2, j);
					m_Features[NumFeatures++] = cTemp;
					sprintf(cTemp, "[%d_%d]-28[%d+]", i, j2, j);
					m_Features[NumFeatures++] = cTemp;
					sprintf(cTemp, "[%d_%d]-46[%d+]", i, j2, j);	//-28 - 18
					m_Features[NumFeatures++] = cTemp;
					sprintf(cTemp, "[%d_%d]-64[%d+]", i, j2, j);	//-28 - 18
					m_Features[NumFeatures++] = cTemp;

				}
			}
		}
	}

	//single amino acid
	char aa[21] = "CDEFHJKLMNOPQRSTUVWY";	//do not include I, and G, A
	for (i = 0; i < 20; i++)	//do not include sty
	{
		sprintf(cTemp, "%c[1+]", aa[i]);	//immonium
		m_Features[NumFeatures++] = cTemp;
		sprintf(cTemp, "%c'[1+]", aa[i]);	//' stands for the acyl form
		m_Features[NumFeatures++] = cTemp;
	}

	m_Features[NumFeatures++] = "K-17[1+]";
	m_Features[NumFeatures++] = "pyroE[1+]";

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