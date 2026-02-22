
#include "CMS2Spectrum.h"

#define PROTON 1.0072765
#define H2O 18.010565
#define NH3	17.026549
#define CO	27.994915
#define DELTA_ISOTOPE 1.0026

CMS2Spectrum::CMS2Spectrum(long n, float *mass, float *intensity, std::string sequence, int charge, float CollisionEnergy, float precursor, CFeatureList *FeatureList, float *FeatureIntensity)
{
	using namespace std;

	if (n > 0 && mass != NULL && intensity != NULL)	// converts spectrum to features (annotations)
	{
		m_points = n;
		m_mass = new float[m_points];
		m_intensity = new float[m_points];
		m_labelscore = new float[m_points];
		m_LabelIntensity = new float[m_points];
		memcpy(m_mass, mass, n * sizeof(float));
		memcpy(m_intensity, intensity, n * sizeof(float));
		x = NULL;
		y = NULL;

		m_label = new string[m_points];
		m_Is4Annotation = 1;
	}
	else
	{
		//convert features to spectrum
		m_points = 0;
		m_mass = new float[131072];	//max spectrum size 128k enough?
		m_intensity = new float[131072];
		x = new float[131072];	//temp space to store the combined spectrum
		y = new float[131072];
		m_labelscore = NULL;
		m_LabelIntensity = NULL;
		m_label = NULL;
		m_Is4Annotation = 0;
	}

	m_FeatureList = FeatureList;
	m_FeatureIntensity = FeatureIntensity;

	//initialize spectrum properties
	m_model = 0;
	m_FragmentMethod = 0;
	m_charge = charge;
	m_CollisionEnergy = CollisionEnergy;
	m_Resolution = 30000.0f;
	m_IsolationWidth = 1.6f;
	m_ReactionTime = 0.0001f;
	m_ActivationQ = 0.0f;
	m_PrecursorMz = precursor;
	m_sequence = sequence;
	m_Mass0 = 0.0;

	m_StartMass = 0.0f;
	m_EndMass = 0.0f;
}

CMS2Spectrum::~CMS2Spectrum()
{
	if (m_mass != NULL)
		delete[] m_mass;
	if (m_intensity != NULL)
		delete[] m_intensity;
	if (x != NULL)
		delete[] x;
	if (y != NULL)
		delete[] y;
	if (m_labelscore != NULL)
		delete[] m_labelscore;
	if (m_label != NULL)
		delete[] m_label;
	if (m_LabelIntensity != NULL)
		delete[] m_LabelIntensity;
}


void CMS2Spectrum::SetFeatures(std::string sequence, int charge, float CollisionEnergy, float precursor, CFeatureList* FeatureList, float* FeatureIntensity)
{
	using namespace std;

	//convert features to spectrum
	m_points = 0;

	m_FeatureList = FeatureList;
	m_FeatureIntensity = FeatureIntensity;

	//initialize spectrum properties
	m_model = 0;
	m_FragmentMethod = 0;
	m_charge = charge;
	m_CollisionEnergy = CollisionEnergy;
	m_Resolution = 30000.0f;
	m_IsolationWidth = 1.6f;
	m_ReactionTime = 0.0001f;
	m_ActivationQ = 0.0f;
	m_PrecursorMz = precursor;
	m_sequence = sequence;
	m_Mass0 = 0.0;
}

void CMS2Spectrum::Annotate()
{
	int length = (int)m_sequence.length();
	double* ResidueMass = new double[length];
	m_Mass0 = GetResidueMasses(length, m_sequence, ResidueMass);

	//make the best guess of Start and End mass (not used yet)
	m_StartMass = (float)(int)(m_mass[0] - 1.0f);
	m_EndMass = (float)(int)(m_mass[m_points - 1] + 1.0f);
	float EndMass0 = (float)(int)(m_Mass0 + 7.0);
	if (m_EndMass > EndMass0 || (m_EndMass < EndMass0 && EndMass0 < 2000.0f))
		m_EndMass = EndMass0;
	if (m_EndMass < 2000.0f && EndMass0 > 2000.0f)
		m_EndMass = 2000.0f;

	//initialize
	int i, j;
	for (i = 0; i < m_points; i++)
	{
		m_labelscore[i] = 0.0f;
		m_label[i] = "";
		m_LabelIntensity[i] = 0.0f;
	}

	double bMass = 0.0;
	float likelihood;
	int H2Oloss = 0, NH3loss = 1, MetOx = 0, NumU = 0, NumJ = 0, PhosST = 0, PhosY = 0;
	int H2Oloss0 = 1, NH3loss0 = 1, MetOx0 = 0, NumU0 = 0, NumJ0 = 0, PhosST0 = 0, PhosY0 = 0;
	for (i = 0; i < length; i++)
	{
		if (m_sequence[i] == 'S' || m_sequence[i] == 'T' || m_sequence[i] == 'D' || m_sequence[i] == 'E')
			H2Oloss0++;
		else if (m_sequence[i] == 'N' || m_sequence[i] == 'Q' || m_sequence[i] == 'K' || m_sequence[i] == 'R')
			NH3loss0++;
		else if (m_sequence[i] == 'O')
			MetOx0++;
		else if (m_sequence[i] == 'U')
			NumU0++;
		else if (m_sequence[i] == 'J')
			NumJ0++;
		else if (m_sequence[i] == 's' || m_sequence[i] == 't')
			PhosST0++;
		else if (m_sequence[i] == 'y')
			PhosY0++;
	}

	//0: check molecular ion
	char FragName[20];	//17 should be enough
	sprintf(FragName, "Mol");
	CheckMass(FragName, m_Mass0, 1.0f, m_charge, 0.002f);	//b ion

	if (NH3loss0)
	{
		sprintf(FragName, "Mol-17");
		CheckMass(FragName, m_Mass0 - NH3, 1.0f, m_charge, 0.002f);
	}

	if (H2Oloss0)
	{
		sprintf(FragName, "Mol-18");
		CheckMass(FragName, m_Mass0 - H2O, 1.0f, m_charge, 0.002f);
	}

	if (MetOx0)
	{
		sprintf(FragName, "Mol-64");
		CheckMass(FragName, m_Mass0 - 63.9983, 1.0f, m_charge, 0.002f);
	}

	if (NumU0)
	{
		sprintf(FragName, "Mol-91");
		CheckMass(FragName, m_Mass0 - 91.009184, 0.8f, m_charge, 0.002f);
	}

	if (NumJ0)
	{
		sprintf(FragName, "Mol-92");
		CheckMass(FragName, m_Mass0 - 91.993200, 0.8f, m_charge, 0.002f);
	}

	if (PhosY0)
	{
		sprintf(FragName, "Mol-80");
		CheckMass(FragName, m_Mass0 - 79.966330, 1.0f, m_charge, 0.002f);
	}

	if (PhosST0 + PhosY0)
	{
		sprintf(FragName, "Mol-98");
		CheckMass(FragName, m_Mass0 - 97.976895, 1.0f, m_charge, 0.002f);
	}

	//other neutral losses
	for (j = OFFSET_FROM; j <= OFFSET_TO; j++)
	{
		if (j!= 64 || !MetOx0)	//j=64 has been checked if MetOx0 is true
		{
			sprintf(FragName, "Mol-%d", j);
			CheckMass(FragName, m_Mass0 - j * 1.0005, 0.5f, m_charge, 0.05f);
		}
	}

	//1: check b and y ions and neutral losses
	int bMaxCharge, yMaxCharge;
	for (i = 0; i < length-1; i++)
	{
		if (m_sequence[i] == 'S' || m_sequence[i] == 'T' || m_sequence[i] == 'D' || m_sequence[i] == 'E')
			H2Oloss++;
		else if (m_sequence[i] == 'N' || m_sequence[i] == 'Q' || m_sequence[i] == 'K' || m_sequence[i] == 'R')
			NH3loss++;
		else if (m_sequence[i] == 'O')
			MetOx++;
		else if (m_sequence[i] == 'U')
			NumU++;
		else if (m_sequence[i] == 'J')
			NumJ++;
		else if (m_sequence[i] == 's' || m_sequence[i] == 't')
			PhosST++;
		else if (m_sequence[i] == 'y')
			PhosY++;

		bMaxCharge = (i + 1) / 2;
		if (bMaxCharge < 1)
			bMaxCharge = 1;
		else if (bMaxCharge > m_charge)
			bMaxCharge = m_charge;
		yMaxCharge = (length - i - 1) / 2;
		if (yMaxCharge < 1)
			yMaxCharge = 1;
		else if (yMaxCharge > m_charge)
			yMaxCharge = m_charge;

		likelihood = CleavageLikelihood(i);	//b/y likelihood 0.9 to 1.1
		bMass += ResidueMass[i];
		sprintf(FragName, "b%d", i + 1);
		CheckMass(FragName, bMass, likelihood, bMaxCharge, 0.002f);	//b ion
		
		sprintf(FragName, "y%d", length - i - 1);
		CheckMass(FragName, m_Mass0 - bMass, likelihood, yMaxCharge, 0.002f);	//y ion

		if (NH3loss)
		{
			sprintf(FragName, "b%d-17", i+1);
			CheckMass(FragName, bMass - NH3, likelihood*0.95f, bMaxCharge, 0.002f);	//b ion
		}
		if (NH3loss0 - NH3loss)
		{
			sprintf(FragName, "y%d-17", length - i - 1);
			CheckMass(FragName, m_Mass0 - bMass - NH3, likelihood * 0.95f, yMaxCharge, 0.002f);	//y ion
		}

		if (H2Oloss)
		{
			sprintf(FragName, "b%d-18", i + 1);
			CheckMass(FragName, bMass - H2O, likelihood * 0.95f, bMaxCharge, 0.002f);	//b ion
		}
		if (H2Oloss0 - H2Oloss)
		{
			sprintf(FragName, "y%d-18", length - i - 1);
			CheckMass(FragName, m_Mass0 - bMass - H2O, likelihood * 0.95f, yMaxCharge, 0.002f);	//y ion
		}

		// a ion
		sprintf(FragName, "b%d-28", i + 1);
		CheckMass(FragName, bMass - CO, likelihood * 0.95f, bMaxCharge, 0.002f);	//a ion

		if (MetOx)
		{
			sprintf(FragName, "b%d-64", i + 1);
			CheckMass(FragName, bMass - 63.9983, likelihood, bMaxCharge, 0.002f);	//b ion
		}
		if (MetOx0 - MetOx)
		{
			sprintf(FragName, "y%d-64", length - i - 1);
			CheckMass(FragName, m_Mass0 - bMass - 63.9983, likelihood, yMaxCharge, 0.002f);	//y ion
		}

		if (NumU)
		{
			sprintf(FragName, "b%d-91", i + 1);
			CheckMass(FragName, bMass - 91.009184, likelihood*0.9f, bMaxCharge, 0.002f);	//b ion
		}
		if (NumU0 - NumU)
		{
			sprintf(FragName, "y%d-91", length - i - 1);
			CheckMass(FragName, m_Mass0 - bMass - 91.009184, likelihood*0.9f, yMaxCharge, 0.002f);	//y ion
		}

		if (NumJ)
		{
			sprintf(FragName, "b%d-92", i + 1);
			CheckMass(FragName, bMass - 91.993200, likelihood * 0.9f, bMaxCharge, 0.002f);	//b ion
		}
		if (NumJ0 - NumJ)
		{
			sprintf(FragName, "y%d-92", length - i - 1);
			CheckMass(FragName, m_Mass0 - bMass - 91.993200, likelihood * 0.9f, yMaxCharge, 0.002f);	//y ion
		}

		if (PhosY)
		{
			sprintf(FragName, "b%d-80", i + 1);
			CheckMass(FragName, bMass - 79.966330, likelihood, bMaxCharge, 0.002f);	//b ion
		}
		if (PhosY0 - PhosY)
		{
			sprintf(FragName, "y%d-80", length - i - 1);
			CheckMass(FragName, m_Mass0 - bMass - 79.966330, likelihood, yMaxCharge, 0.002f);	//y ion
		}

		if (PhosST + PhosY)
		{
			sprintf(FragName, "b%d-98", i + 1);
			CheckMass(FragName, bMass - 97.976895, likelihood, bMaxCharge, 0.002f);	//b ion
		}
		if (PhosST0 + PhosY0 - PhosST - PhosY)
		{
			sprintf(FragName, "y%d-98", length - i - 1);
			CheckMass(FragName, m_Mass0 - bMass - 97.976895, likelihood, yMaxCharge, 0.002f);	//y ion
		}

		//additions to form c, x ions
		sprintf(FragName, "b%d+17", i + 1);
		CheckMass(FragName, bMass + NH3, likelihood * 0.8f, bMaxCharge, 0.002f);	//c ion

		sprintf(FragName, "b%d+18", i + 1);
		CheckMass(FragName, bMass + H2O, likelihood * 0.8f, bMaxCharge, 0.002f);	//bn-1 + H2O C-term rearrangement

		sprintf(FragName, "y%d+28", length - i - 1);
		CheckMass(FragName, m_Mass0 - bMass + CO, likelihood * 0.5f, yMaxCharge, 0.002f);	//x ion

		//other neutral losses
		for (j = OFFSET_FROM; j <= OFFSET_TO; j++)
		{
			if (j != 64 || !MetOx)
			{
				sprintf(FragName, "b%d-%d", i + 1, j);
				CheckMass(FragName, bMass - j * 1.0005, likelihood * 0.5f, bMaxCharge, 0.05f);	//b ion
			}

			if (j != 64 || !(MetOx0 - MetOx))
			{
				sprintf(FragName, "y%d-%d", length - i - 1, j);
				CheckMass(FragName, m_Mass0 - bMass - j * 1.0005, likelihood * 0.5f, yMaxCharge, 0.05f);	//y ion
			}
		}
	}

	//internal fragments (at least two residues)
	double fragmass;
	float likelihood1;
	for (i = 0; i < length - 3; i++)	//first cut (on the right side of i)
	{
		fragmass = 0.0;
		likelihood1 = CleavageLikelihood(i);
		for (j = i + 1; j < length - 1; j++)	//second cut
		{
			fragmass += ResidueMass[j];
			if (j > i + 1)	//at least two residues
			{
				bMaxCharge = (j - i) * 2 / 5;	//match criteria in CFeatureList for internal fragment
				if (bMaxCharge < 1)
					bMaxCharge = 1;
				else if (bMaxCharge > m_charge)
					bMaxCharge = m_charge;
				likelihood = likelihood1 * CleavageLikelihood(j) * 0.75f;
				sprintf(FragName, "[%d_%d]", i+2, j+1);
				CheckMass(FragName, fragmass, likelihood, bMaxCharge, 0.002f);
				sprintf(FragName, "[%d_%d]-17", i + 2, j + 1);
				CheckMass(FragName, fragmass - NH3, likelihood * 0.9f, bMaxCharge, 0.002f);
				sprintf(FragName, "[%d_%d]-18", i + 2, j + 1);
				CheckMass(FragName, fragmass - H2O, likelihood * 0.9f, bMaxCharge, 0.002f);
				sprintf(FragName, "[%d_%d]-28", i + 2, j + 1);
				CheckMass(FragName, fragmass-CO, likelihood*0.9f, bMaxCharge, 0.002f);	//internal immonium
			}
		}
	}

	//single amino acid
	int skip = 0;
	for (i = 0; i < length; i++)
	{
		skip = 0;
		for (j = 0; j < i; j++)
		{
			if (m_sequence[i] == m_sequence[j])
			{
				skip = 1;
				break;
			}
		}
		if (!skip)
		{
			sprintf(FragName, "%c'", m_sequence[i]);	//amino acid acyl
			CheckMass(FragName, ResidueMass[i], 0.5f, 1, 0.002f);
			sprintf(FragName, "%c", m_sequence[i]);	//amino acid immonium
			CheckMass(FragName, ResidueMass[i] - CO, 0.9f, 1, 0.002f);
		}
	}
	delete[] ResidueMass;
}



int CMS2Spectrum::CheckMass(char FragName[], double FragMass, float likelihood, int MaxCharge, float MinTolerance)
{
	int z1 = 1;
	if (strstr(FragName, "Mol") == FragName)
		z1 = MaxCharge;
	int z2 = MaxCharge;
	int z;
	float accuracy, accuracy0 = (float)FragMass*1.0e-5f;	//10 ppm
	if (accuracy0 < MinTolerance)
		accuracy0 = MinTolerance;
	float accuracy_isotope, accuracy_isotope0 = accuracy0;
	if (MinTolerance > 0.02f)
	{
		accuracy_isotope0 = (float)FragMass * 1.0e-5f;	//10 ppm;
		if (accuracy_isotope0 < 0.003f)
			accuracy_isotope0 = 0.003f;
	}

	//calculate isotope distribution
	float a[64], a2[64];	//isotope distribution. a[1] is monoisotopic
	int index[64];		//index of spectrum for each isotope in a2[]
	int i, j, top, top_index;
	char NewLabel[20];
	int n = IsotopeSimulationFromMass(FragMass, (int)(7.0 + FragMass/1000.0), a, 0.08f, &top);	//top is the index of top of a[]
	
	if (m_Is4Annotation)
	{
		float max = a[top];
		double mz, topmz;
		float currentmz;
		int topindex_real, index1_real, index2_real, top2;	//indices for real spectrum and a[]
		float sum1, sumxy1, sum2, sumxy2, product, score;
		float FeatureIntensity;

		for (z = z1; z <= z2; z++)
		{
			accuracy = accuracy0 / z;
			accuracy_isotope = accuracy_isotope0 / z;
			mz = FragMass / z + PROTON;	//monoisotopic m/z
			topmz = (FragMass + DELTA_ISOTOPE * (top - 1)) / z + PROTON;

			if (topmz < m_StartMass || topmz > m_EndMass)	//set feature intensity to -1 if outside the scan range
			{
				if (strstr(FragName, "Mol") == FragName)	//do not include charge for molecular ions
					sprintf(NewLabel, "%s", FragName);
				else
					sprintf(NewLabel, "%s[%d+]", FragName, z);

				if (NewLabel[0] == 'I')	//I or I' is not in the dictionary
					NewLabel[0] = 'L';

				//search NewLabel in the feature list
				for (i = 0; i < m_FeatureList->m_NumFeatures; i++)
				{
					if (m_FeatureList->m_Features[i] == NewLabel)
					{
						m_FeatureIntensity[i] = -1.0f;
						break;
					}
				}
				continue;
			}
			if (topmz < m_mass[0] - DELTA_ISOTOPE / z - accuracy || topmz > m_mass[m_points - 1] + DELTA_ISOTOPE / z + accuracy)
				continue;

			for (i = 0; i < n; i++)
			{
				a2[i] = 0.0f;	//experimental
				index[i] = -1;
			}

			//find top
			top_index = -1;	//index of top of m_intensity[]
			index1_real = -1;
			index2_real = -1;
			for (i = 0; i < m_points && m_mass[i] < topmz + accuracy; i++)
			{
				if (m_mass[i] > topmz - accuracy)
				{
					if (top_index < 0 || m_intensity[i] > m_intensity[top_index])
						top_index = i;
				}
			}
			if (top_index >= 0)	//replace topmz with experimental value
			{
				topmz = m_mass[top_index];
				a2[top] = m_intensity[top_index];
			}
			else
				continue;

			//store all other isotopic peaks in a2[]
			topindex_real = top_index;	//real top in a2[] may be different from in a[]
			top2 = top;
			i = top_index - 1;	//index in the spectrum
			if (i < 0)
				i = 0;

			j = top + 1;	//index in a[]
			while (j < n && i < m_points)
			{
				currentmz = (float)(topmz + (j - top) * DELTA_ISOTOPE / z);
				while (i < m_points && m_mass[i] < currentmz - accuracy_isotope)
					i++;
				while (i < m_points && m_mass[i] < currentmz + accuracy_isotope)
				{
					if (a2[j] < m_intensity[i])
					{
						a2[j] = m_intensity[i];
						index[j] = i;
					}
					i++;
				}
				if (index[j] >= 0)
				{
					if (a2[j] > a2[top])
					{
						top2 = j;	//index for peak top in a2[]
						topindex_real = index[j];
					}
					index2_real = index[j];	//index of the last isotope
				}
				else
					break;
				j++;
			}

			i = top_index;
			j = top - 1;
			while (j >= 0 && i >= 0)
			{
				currentmz = (float)(topmz - (top - j) * DELTA_ISOTOPE / z);
				while (i >= 0 && m_mass[i] > currentmz + accuracy_isotope)
					i--;
				while (i >= 0 && m_mass[i] > currentmz - accuracy_isotope)
				{
					if (a2[j] < m_intensity[i])
					{
						a2[j] = m_intensity[i];
						index[j] = i;
					}
					i--;
				}
				if (index[j] >= 0)
				{
					if (a2[j] > a2[top2])
					{
						top2 = j;
						topindex_real = index[j];
					}
					index1_real = index[j];	//index of the first isotope
				}
				else
					break;
				j--;
			}

			//test whether a[] and a2[] are too different before calculating similarity
			if (a[top2] < a[top] * 0.25f || abs(top2 - top) > n / 2)	//experimental peak top should be much different theoretically
				continue;

			//calculate similarity
			sum1 = 0.0f;
			sumxy1 = 0.0f;
			sum2 = 0.0f;
			sumxy2 = 0.0f;
			product = 0.0f;
			for (i = 0; i < n; i++)
			{
				sum1 += a[i];
				sum2 += a2[i];
				product += (float)sqrt((double)a[i] * a2[i]);
				sumxy1 += a[i] * i;
				sumxy2 += a2[i] * i;
			}
			FeatureIntensity = sum2;	//all experimental isotopes summed up

			//use real spectrum to calculate sum2 so that peaks between isotopes will have penalty
			if (index1_real >= 0 && index2_real >= index1_real)
			{
				sum2 = 0.0f;
				for (i = index1_real; i <= index2_real; i++)
					sum2 += m_intensity[i];
			}

			score = product / (float)sqrt((double)(sum1 * sum2)) * likelihood;

			//store labeling result
			if (score > 0.25f && score > m_labelscore[topindex_real] * 0.8f)
			{
				if (strstr(FragName, "Mol") == FragName)	//do not include charge for molecular ions
					sprintf(NewLabel, "%s\n", FragName);
				else
					sprintf(NewLabel, "%s[%d+]\n", FragName, z);
				if (score > m_labelscore[topindex_real] * 1.25f)//remove old label and replace with the new one
				{
					m_label[topindex_real] = NewLabel;
					m_labelscore[topindex_real] = score;
					m_LabelIntensity[topindex_real] = FeatureIntensity;
				}
				else if (score > m_labelscore[topindex_real])
				{
					m_label[topindex_real] = NewLabel + m_label[topindex_real];
					m_labelscore[topindex_real] = score;
					m_LabelIntensity[topindex_real] = FeatureIntensity;
				}
				else
				{
					m_label[topindex_real] += NewLabel;
				}

				//apply new score to all isotopes
				for (i = 1; i < n; i++)	//i=1 is the monoisotopic peak
				{
					if (index[i] >= 0 && score > m_labelscore[index[i]])
					{
						m_labelscore[index[i]] = score;
					}
				}
			}
		}
	}
	else
	{
		float x[32], y[32]; //one isotope envelop to be added
		//Generate isotope envelop and then add to spectrum
		for (z = z1; z <= z2; z++)
		{
			if (strstr(FragName, "Mol") == FragName)	//do not include charge for molecular ions
				sprintf(NewLabel, "%s", FragName);
			else
				sprintf(NewLabel, "%s[%d+]", FragName, z);

			for (i = 0; i < m_FeatureList->m_NumFeatures; i++)
			{
				if (m_FeatureList->m_Features[i] == NewLabel)
				{
					//Add to spectrum
					if (m_FeatureIntensity[i] > 0.0f)
					{
						for (j = 0; j < n - 1; j++)
						{
							x[j] = (float)((FragMass + j * DELTA_ISOTOPE) / z + PROTON);
							y[j] = a[j + 1] * m_FeatureIntensity[i];	//a[1] is monoisotopic
						}
						AddToSpectrum(n - 1, x, y);
					}
					break;
				}
			}
		}
	}
	return 0;
}

float CMS2Spectrum::CleavageLikelihood(int CleavageSite)
{
	float likelihood = 1.0f;
	if (m_sequence[CleavageSite + 1] == 'P' || CleavageSite == 1)	//X-P or b2
		likelihood += 0.1f;
	if (m_sequence[CleavageSite] == 'P' || m_sequence[CleavageSite] == 'G')
		likelihood -= 0.1f;
	else if (m_sequence[CleavageSite] == 'V' || m_sequence[CleavageSite] == 'I' || m_sequence[CleavageSite] == 'L')
		likelihood += 0.05f;

	return likelihood;
}

//get mass of each residue, return peptide mass
double CMS2Spectrum::GetResidueMasses(int length, std::string sequence, double *ResidueMass)
{
	int i, j;
	double mass = H2O;
	char aa[27] = "ACDEFGHIJKLMNOPQRSTUVWYsty";
	double aamass[26] = { 71.037114,103.009184,115.026943,129.042593,147.068414,57.021464,137.058912,113.084064,161.014663786,128.094963,113.084064,131.040485,114.042927,147.035399,97.052764,128.058578,156.101111,87.032028,101.047678,160.030648,99.068414,186.079313,163.063329,166.998358,181.014008,243.029658 };

	for (i = 0; i < length; i++)
	{
		for (j = 0; j < 26; j++)
		{
			if (sequence[i] == aa[j])
			{
				ResidueMass[i] = aamass[j];
				mass += aamass[j];
				break;
			}
		}
	}
	return mass;
}

//quickly estimate isotope distribution of peptide with mass of FragMass
//to do: can we use binomial distribution with only two parameters p[0] (isotope abundance) and n = M/p[1]?
int CMS2Spectrum::IsotopeSimulationFromMass(double mass, int size, float a[], float cutoff, int *top)
{
	double p[16] = {3.8097,1.6618,1.8358,-11.12,0.9072,-0.3377,2.1532,-0.7235,0.5955,5.6631,0.0875,-72.75,15.29,702.64,5.887,0.8525};
	double MassE4 = 1.0e-4 * mass;

	double m1 = p[12] / sqrt(mass);
	double m2 = p[13]/mass;
	double m3 = p[14]*MassE4;

	int n;
	a[0] = 0.0f;
	for (n = 0; n < size - 1; n++)
		a[n + 1] = (float)(m1 * exp(-m2 * (n - m3) * (n - m3)));

	if (MassE4 <= p[15])
	{
		m1 = p[0]*exp(p[1]*MassE4);
		m2 = p[2] * exp(p[3] * MassE4) + p[4] + p[5] * MassE4;
		m3 = p[6]*exp(p[7]*MassE4);
		double m4 = p[8] + p[9] * MassE4 + p[10] * exp(p[11] * MassE4);
		for (n = 0; n < size - 1; n++)
			a[n + 1] = (float)(MassE4 / p[15] * a[n + 1] + (p[15] - MassE4) / p[15] * exp(-m1 / (n + m3) - m2 * (n - m4)));
	}

	//truncate and normalize
	float max = 0.0f, sum = 0.0f;
	n = 1;
	*top = 1;
	while (n < size && a[n] >= cutoff * max)
	{
		sum += a[n];
		if (max < a[n])
		{
			max = a[n];
			*top = n;
		}
		n++;
	}
	
	int i;
	for (i = 0; i < n; i++)
		a[i] /= sum;

	return n;
}

//export features for this spectrum
void CMS2Spectrum::ExportAnnotations(char filename[])
{
	using namespace std;
	ofstream file(filename, ios::app);	//append to file

	//sequence
	file << '>' << "length=" << m_sequence.length() << "; sequence=" << m_sequence << "; charge=" << m_charge << "; NCE=" << (int)(m_CollisionEnergy+0.5f) << '\n';

	//annotations
	int j, k, iStart, b, success;
	char OneLabel[20];

	for (j = 0; j < m_points; j++)
	{
		if (m_label[j].length() > 0)
		{
			iStart = 0;
			if ((b = (int)m_label[j].find('\n', iStart)) > 0)	//Export only the top label. Change if to while if export other possible labels for this m/z (causing problem due to redundent intensity prediction)
			{
				//a label is from iStart to b
				m_label[j].copy(OneLabel, b - iStart, iStart);
				OneLabel[b - iStart] = '\0';

				if (OneLabel[0] == 'I')	//I or I' is not in the dictionary
					OneLabel[0] = 'L';

				//search OneLabel in the feature list
				success = 0;
				for (k = 0; k < m_FeatureList->m_NumFeatures; k++)
				{
					if (m_FeatureList->m_Features[k] == OneLabel)
					{
						m_FeatureIntensity[k] = m_LabelIntensity[j];
						success = 1;
						break;
					}
				}
				iStart = b + 1;

				if (!success)
				{
					//record unrecognized labels
				}
			}
		}
	}
	
	//save annotations
	for (k = 0; k < m_FeatureList->m_NumFeatures; k++)
	{
		if (m_FeatureIntensity[k] > 0.5f)
		{
//			file << k << ',' << m_FeatureList->m_Features[k] << ',' << (int)(m_FeatureIntensity[k] + 0.5f) << '\n';
			file << k << ',' << (int)(m_FeatureIntensity[k] + 0.5f) << '\n';
		}
		else if (m_FeatureIntensity[k] < 0.0f)
			file << k << ',' << m_FeatureIntensity[k] << '\n';
	}
	file.close();
}


//export features for this spectrum
//not used right now
void CMS2Spectrum::ExportSpectrum(std::ofstream file)
{
	int i;

	//normalize to total signal of 1e7
	float sum = 0.0f;
	for (i = 0; i < m_points; i++)
		sum += m_intensity[i];

	file << '>' << "length=" << m_sequence.length() << "; sequence=" << m_sequence << "; charge=" << m_charge << "; NCE=" << (int)(m_CollisionEnergy + 0.5f) << '\n';
	file << m_points << '\n';
	for (i = 0; i < m_points; i++)
		file << m_mass[i] << ',' << m_intensity[i]/sum*1.0e7f << '\n';
}

//Add the isotope envelop to the spectrum
//x and y is a temp space to store temp spectrum
int CMS2Spectrum::AddToSpectrum(int n2, float x2[], float y2[])
{
	float tol = x2[0] * 1.0e-5f;	//10 ppm or 0.002, whichever is larger
	if (tol < 0.002f)
		tol = 0.002f;

	//new spectrum has larger m/z
	if (m_points == 0 || x2[0] > m_mass[m_points - 1] + tol)
	{
		memcpy(m_mass + m_points, x2, n2 * sizeof(float));
		memcpy(m_intensity + m_points, y2, n2 * sizeof(float));
		return m_points = m_points + n2;
	}

	//new spectrum has smaller m/z
	if (m_mass[0] > x2[n2 - 1] + tol)
	{
		memcpy(x, m_mass, m_points * sizeof(float));
		memcpy(y, m_intensity, m_points * sizeof(float));
		memcpy(m_mass, x2, n2 * sizeof(float));
		memcpy(m_intensity, y2, n2 * sizeof(float));
		memcpy(m_mass + n2, x, m_points * sizeof(float));
		memcpy(m_intensity + n2, y, m_points * sizeof(float));
		return m_points = m_points + n2;
	}

	//overlapping m/z
	int pointer1 = 0;
	if (x2[0] > m_mass[0])
	{
		int next = m_points - 1;
		GoldenSearch(&pointer1, &next, x2[0]);
	}

	int start = pointer1;
	int pointer2 = 0;

	int index = 0;
	while (pointer1 < m_points && pointer2 < n2)
	{
		if (x2[pointer2] - m_mass[pointer1] > tol)
		{
			x[index] = m_mass[pointer1];
			y[index] = m_intensity[pointer1];
			index++;
			pointer1++;
		}
		else if (m_mass[pointer1] - x2[pointer2] > tol)
		{
			x[index] = x2[pointer2];
			y[index] = y2[pointer2];
			index++;
			pointer2++;
		}
		else
		{
			while (pointer1 + 1 < m_points && fabs(m_mass[pointer1 + 1] - x2[pointer2]) < fabs(m_mass[pointer1] - x2[pointer2]))
			{
				x[index] = m_mass[pointer1];
				y[index] = m_intensity[pointer1];
				index++;
				pointer1++;
			}

			if (m_intensity[pointer1] + y2[pointer2] > 0.0f)
				x[index] = (m_mass[pointer1] * m_intensity[pointer1] + x2[pointer2] * y2[pointer2]) / (m_intensity[pointer1] + y2[pointer2]);
			else
				x[index] = m_mass[pointer1];
			y[index] = m_intensity[pointer1] + y2[pointer2];
			index++;
			pointer1++;
			pointer2++;
		}
	}

	if (pointer1 == m_points)
	{
		memcpy(x + index, x2 + pointer2, (n2 - pointer2) * sizeof(float));
		memcpy(y + index, y2 + pointer2, (n2 - pointer2) * sizeof(float));
		index += (n2 - pointer2);
	}
	else if (pointer2 == n2)
	{
		memcpy(x + index, m_mass + pointer1, (m_points - pointer1) * sizeof(float));
		memcpy(y + index, m_intensity + pointer1, (m_points - pointer1) * sizeof(float));
		index += (m_points - pointer1);
	}

	//copy x[] to m_mass[] from start
	memcpy(m_mass + start, x, index * sizeof(float));
	memcpy(m_intensity + start, y, index * sizeof(float));
	m_points = start + index;
	return m_points;
}

//using golden-fraction method to search the index of mass0
//initialize from and to as the first and last points
void CMS2Spectrum::GoldenSearch(int* from, int* to, float mass0)
{
	if (mass0 <= m_mass[*from])
	{
		*to = *from + 1;
		return;
	}
	else if (mass0 >= m_mass[*to])
	{
		*from = *to - 1;
		return;
	}

	int bracket;
	while (*to - *from > 1)
	{
		bracket = *from + 1 + (int)((*to - *from - 1) * 0.618f);

		if (mass0 < m_mass[bracket])
			*to = bracket;
		else if (mass0 > m_mass[bracket])
			*from = bracket;
		else
		{
			if (bracket + 1 <= *to)
			{
				*from = bracket;
				*to = bracket + 1;
			}
			else
			{
				*from = bracket - 1;
				*to = bracket;
			}
			break;
		}
	}
	return;
}