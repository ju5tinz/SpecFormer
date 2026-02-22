
#include <algorithm>
#include "CMS2Spectrum.h"

#define PROTON 1.0072765
#define H2O 18.010565
#define NH3	17.026549
#define CO	27.994915
#define DELTA_ISOTOPE 1.0026

//likelihood score based on the frequency of occurance of a fragment ion (relative to ~1.0 of b/y ions)
#define VERYCOMMON	1.0f
#define COMMON	0.9f
#define LESSCOMMON	0.8f
#define RARE	0.7f
#define VERYRARE	0.5f

extern int bOffset[];
extern int yOffset[];

float combin[16][16] =
{
	1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,
	0.0f,1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,9.0f,10.0f,11.0f,12.0f,13.0f,14.0f,15.0f,
	0.0f,0.0f,1.0f,3.0f,6.0f,10.0f,15.0f,21.0f,28.0f,36.0f,45.0f,55.0f,66.0f,78.0f,91.0f,105.0f,
	0.0f,0.0f,0.0f,1.0f,4.0f,10.0f,20.0f,35.0f,56.0f,84.0f,120.0f,165.0f,220.0f,286.0f,364.0f,455.0f,
	0.0f,0.0f,0.0f,0.0f,1.0f,5.0f,15.0f,35.0f,70.0f,126.0f,210.0f,330.0f,495.0f,715.0f,1001.0f,1365.0f,
	0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,6.0f,21.0f,56.0f,126.0f,252.0f,462.0f,792.0f,1287.0f,2002.0f,3003.0f,
	0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,7.0f,28.0f,84.0f,210.0f,462.0f,924.0f,1716.0f,3003.0f,5005.0f,
	0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,8.0f,36.0f,120.0f,330.0f,792.0f,1716.0f,3432.0f,6435.0f,
	0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,9.0f,45.0f,165.0f,495.0f,1287.0f,3003.0f,6435.0f,
	0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,10.0f,55.0f,220.0f,715.0f,2002.0f,5005.0f,
	0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,11.0f,66.0f,286.0f,1001.0f,3003.0f,
	0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,12.0f,78.0f,364.0f,1365.0f,
	0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,13.0f,91.0f,455.0f,
	0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,14.0f,105.0f,
	0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f,15.0f,
	0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,1.0f
};

CMS2Spectrum::CMS2Spectrum(long n, float *mass, float *intensity, std::string sequence, int charge, float CollisionEnergy, float precursor, CFeatureList *FeatureList, float *FeatureIntensity, float StartMass, float EndMass)
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

		m_NumFragIons = 0;
		m_FragIonMz = NULL;
		m_FragIonCharge = NULL;
		m_FragIonIntensity = NULL;
		m_SortIndex = NULL;

		int i;
		m_MaxSignal = 0.0f;
		for (i = 0; i < m_points; i++)
		{
			if (m_MaxSignal < m_intensity[i])
				m_MaxSignal = m_intensity[i];
		}
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
		m_MaxSignal = 0.0f;

		//space for storing fragment ion information before adding to the predicted spectrum
		m_NumFragIons = 0;
		m_FragIonMz = new float[FeatureList->m_NumFeatures];
		m_FragIonCharge = new int[FeatureList->m_NumFeatures];
		m_FragIonIntensity = new float[FeatureList->m_NumFeatures];
		m_SortIndex = new int[FeatureList->m_NumFeatures];
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

	m_StartMass = StartMass;
	m_EndMass = EndMass;
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

	if (m_FragIonMz != NULL)
		delete [] m_FragIonMz;
	if (m_FragIonCharge != NULL)
		delete [] m_FragIonCharge;
	if (m_FragIonIntensity != NULL)
		delete [] m_FragIonIntensity;
	if (m_SortIndex != NULL)
		delete[] m_SortIndex;
}


void CMS2Spectrum::SetFeatures(std::string sequence, int charge, float CollisionEnergy, float precursor, CFeatureList* FeatureList, float* FeatureIntensity, float StartMass, float EndMass)
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

	m_StartMass = StartMass;
	m_EndMass = EndMass;
}


void CMS2Spectrum::Annotate()
{
	int length = (int)m_sequence.length();
	double* ResidueMass = new double[length];
	m_Mass0 = GetResidueMasses(length, m_sequence, ResidueMass);	//return peptide mass

	//make the best guess of Start and End mass
	if (m_StartMass == 0.0f)
	{
		m_StartMass = (float)(int)(m_mass[0] - 1.0f);
	}
	if (m_EndMass == 0.0f)
	{
		m_EndMass = (float)(int)(m_mass[m_points - 1] + 1.0f);
		float EndMass0 = (float)(int)(m_Mass0 + 7.0);
		if (m_EndMass > EndMass0 || (m_EndMass < EndMass0 && EndMass0 < 2000.0f))
			m_EndMass = EndMass0;
		if (m_EndMass < 2000.0f && EndMass0 > 2000.0f)
			m_EndMass = 2000.0f;
	}

	//initialize
	m_NumFragIons = 0;	//for constructing spectrum
	int i, j;
	for (i = 0; i < m_points; i++)	//for annotation
	{
		m_labelscore[i] = 0.0f;
		m_label[i] = "";
		m_LabelIntensity[i] = 0.0f;
	}

	double bMass = 0.0;
	float likelihood;
	int H2Oloss = 0, NH3loss = 1, MetOx = 0, NumU = 0, NumJ = 0, PhosST = 0, PhosY = 0, Arg = 0;
	int H2Oloss0 = 1, NH3loss0 = 1, MetOx0 = 0, NumU0 = 0, NumJ0 = 0, PhosST0 = 0, PhosY0 = 0, Arg0 = 0;
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
		else if (m_sequence[i] == 'R')
			Arg0++ ;
	}

	//bIntensity and yIntensity are used to record b/y ion intensity of different charges
	//they are used to determine whether to search for the minor neutral losses and how to score them
	int k;
	float bIntensity[8], yIntensity[8];	//maximum 7 charge
	for (k = 0; k < 8; k++)
	{
		bIntensity[k] = 0.0f;
	}

	//0: check molecular ion
	char FragName[20];	//17 should be enough
	sprintf(FragName, "Mol");
	CheckFragment(FragName, m_Mass0, VERYCOMMON, 1, bIntensity, 0.005f);	//b ion

	if (NH3loss0)
	{
		sprintf(FragName, "Mol-17");
		CheckFragment(FragName, m_Mass0 - NH3, VERYCOMMON, 1, bIntensity, 0.005f);
	}

	if (H2Oloss0)
	{
		sprintf(FragName, "Mol-18");
		CheckFragment(FragName, m_Mass0 - H2O, VERYCOMMON, 1, bIntensity, 0.005f);
	}

	if (MetOx0)
	{
		sprintf(FragName, "Mol-64");
		CheckFragment(FragName, m_Mass0 - 63.998286, VERYCOMMON, 1, bIntensity, 0.005f);
	}

	if (PhosY0)
	{
		sprintf(FragName, "Mol-80");
		CheckFragment(FragName, m_Mass0 - 79.966330, VERYCOMMON, 1, bIntensity, 0.005f);
	}

	if (PhosST0 + PhosY0)
	{
		sprintf(FragName, "Mol-98");
		CheckFragment(FragName, m_Mass0 - 97.976895, VERYCOMMON, 1, bIntensity, 0.005f);
	}

	//other neutral losses
	float OffsetLikelihood;
	for (j = 0; j < NUM_Y_OFFSETS; j++)
	{
		if (yOffset[j] != 64 || !MetOx0)	//j=64 has been checked if MetOx0 is true
		{
			OffsetLikelihood = VERYRARE;
			if (yOffset[j] == 34 && NH3loss0 > 1)	//M - 2NH3
				OffsetLikelihood = RARE;
			else if (yOffset[j] == 35 && NH3loss0 && H2Oloss0)	//M - H2O - NH3
				OffsetLikelihood = LESSCOMMON;
			else if (yOffset[j] == 36 && H2Oloss0 > 1)	//M - 2H2O
				OffsetLikelihood = COMMON;
			else if (yOffset[j] == 51 && NH3loss0 > 2)	//M - 3NH3
				OffsetLikelihood = RARE;
			else if (yOffset[j] == 52 && NH3loss0 > 1 && H2Oloss0)	//M - 2NH3 - H2O
				OffsetLikelihood = RARE;
			else if (yOffset[j] == 53 && NH3loss0 && H2Oloss0 > 1)	//M - NH3 - 2H2O
				OffsetLikelihood = RARE;
			else if (yOffset[j] == 54 && H2Oloss0 > 2)	//M - 3H2O
				OffsetLikelihood = RARE;

			if (yOffset[j] == 57 && (m_sequence[0] == 'G' || m_sequence[length - 1] == 'G'))	//this is y(n-1)
				OffsetLikelihood = 0.0f;
			else if (Arg0 < m_charge && yOffset[j] == 42)// || yOffset[j] == 44 || yOffset[j] == 59 || yOffset[j] == 61) && Arg0 < m_charge)	//R - 42
				OffsetLikelihood = 0.0f;
			else if (yOffset[j] == 60 && (Arg0 < m_charge || m_sequence[length - 1] != 'R'))	//C-term R
				OffsetLikelihood = 0.0f;

			if (OffsetLikelihood > 0.0f)
			{
				sprintf(FragName, "Mol-%d", yOffset[j]);
				CheckFragment(FragName, m_Mass0 - yOffset[j] * 1.0005, OffsetLikelihood, 0, bIntensity, 0.05f);
			}
		}
	}

	//Loss of C-term residue through C-term rearrangement (or called bn-1 + H2O ion)
	sprintf(FragName, "Mol-Cterm");
	CheckFragment(FragName, m_Mass0 - ResidueMass[length - 1], LESSCOMMON, 1, bIntensity, 0.005f);


	//1: check b and y ions and neutral losses
	int bMaxCharge, yMaxCharge;
	for (i = 0; i < length-1; i++)
	{
		for (k = 0; k < 8; k++)
		{
			bIntensity[k] = 0.0f;
			yIntensity[k] = 0.0f;
		}

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
		else if (m_sequence[i] == 'R')
			Arg++;

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

		likelihood = GetCleavageLikelihood(i);	//b/y likelihood 0.9 to 1.1
		bMass += ResidueMass[i];
		sprintf(FragName, "b%d", i + 1);
		CheckFragment(FragName, bMass, likelihood, 1, bIntensity, 0.005f);	//b ion
		
		sprintf(FragName, "y%d", length - i - 1);
		CheckFragment(FragName, m_Mass0 - bMass, likelihood, 1, yIntensity, 0.005f);	//y ion

		//special b2 neutral losses (not in the Dictionary)
		if (i == 1)
		{
			if (m_sequence[i] == 'T')
			{
				sprintf(FragName, "b%d-73", i + 1);
				CheckFragment(FragName, bMass - 73.01638, likelihood * COMMON, 1, bIntensity, 0.005f);	//b2 - C2H3NO2
			}
			else if (m_sequence[i] == 'M')
			{
				sprintf(FragName, "b%d-76", i + 1);
				CheckFragment(FragName, bMass - 75.99829, likelihood * COMMON, 1, bIntensity, 0.005f);	//b2 - C2H4SO
			}
			else if (m_sequence[i] == 'W')
			{
				sprintf(FragName, "b%d-159", i + 1);
				CheckFragment(FragName, bMass - 159.06841, likelihood * COMMON, 1, bIntensity, 0.005f);	//b2 - C10H9NO
			}
			else if (m_sequence[i] == 'U')
			{
				sprintf(FragName, "b%d-143", i + 1);
				CheckFragment(FragName, bMass - 143.00410, likelihood * COMMON, 1, bIntensity, 0.005f);	//b2 - C5H5NO2S
			}
		}

		if (NH3loss)
		{
			sprintf(FragName, "b%d-17", i+1);
			if (m_sequence[0] == 'Q')
				CheckFragment(FragName, bMass - NH3, likelihood, 1, bIntensity, 0.005f);
			else
				CheckFragment(FragName, bMass - NH3, likelihood* COMMON, 1, bIntensity, 0.005f);	//b ion
		}
		if (NH3loss0 - NH3loss)
		{
			sprintf(FragName, "y%d-17", length - i - 1);
			if (m_sequence[i+1] == 'Q')
				CheckFragment(FragName, m_Mass0 - bMass - NH3, likelihood, 1, yIntensity, 0.005f);	//y ion
			else
				CheckFragment(FragName, m_Mass0 - bMass - NH3, likelihood * COMMON, 1, yIntensity, 0.005f);	//y ion
		}

		if (H2Oloss)
		{
			sprintf(FragName, "b%d-18", i + 1);
			if (m_sequence[0] == 'E')
				CheckFragment(FragName, bMass - H2O, likelihood, 1, bIntensity, 0.002f);	//b ion
			else
				CheckFragment(FragName, bMass - H2O, likelihood* COMMON, 1, bIntensity, 0.002f);	//b ion
		}
		else
		{
			sprintf(FragName, "b%d-18", i + 1);
			CheckFragment(FragName, bMass - H2O, likelihood * LESSCOMMON, 0, bIntensity, 0.005f);	//b ion water loss from C-term (less common but possible)
		}

		if (H2Oloss0 - H2Oloss)
		{
			sprintf(FragName, "y%d-18", length - i - 1);
			if (m_sequence[i + 1] == 'E')
				CheckFragment(FragName, m_Mass0 - bMass - H2O, likelihood, 1, yIntensity, 0.005f);
			else
				CheckFragment(FragName, m_Mass0 - bMass - H2O, likelihood * COMMON, 1, yIntensity, 0.005f);	//y ion
		}

		// a ion
		sprintf(FragName, "b%d-28", i + 1);
		if (i <= 1)	//a1 and a2 more likely
			CheckFragment(FragName, bMass - CO, likelihood, 1, bIntensity, 0.005f);	//a ion
		else
			CheckFragment(FragName, bMass - CO, likelihood * LESSCOMMON, 1, bIntensity, 0.005f);	//a ion

		if (MetOx)
		{
			sprintf(FragName, "b%d-64", i + 1);
			CheckFragment(FragName, bMass - 63.998286, likelihood, 1, bIntensity, 0.005f);	//b ion

			if (H2Oloss)	//this is not in the Dictionary
			{
				sprintf(FragName, "b%d-82", i + 1);
				CheckFragment(FragName, bMass - 63.998286 - H2O, likelihood* LESSCOMMON, 0, bIntensity, 0.005f);
			}
			if (NH3loss)	//this is not in the Dictionary
			{
				sprintf(FragName, "b%d-81", i + 1);
				CheckFragment(FragName, bMass - 63.998286 - NH3, likelihood* LESSCOMMON, 0, bIntensity, 0.005f);
			}
		}
		if (MetOx0 - MetOx)
		{
			sprintf(FragName, "y%d-64", length - i - 1);
			CheckFragment(FragName, m_Mass0 - bMass - 63.998286, likelihood, 1, yIntensity, 0.005f);	//y ion

			if (H2Oloss0 - H2Oloss)//this is not in the Dictionary
			{
				sprintf(FragName, "y%d-82", length - i - 1);
				CheckFragment(FragName, m_Mass0 - bMass - 63.998286 - H2O, likelihood* LESSCOMMON, 0, yIntensity, 0.005f);	//y ion
			}
			if (NH3loss0 - NH3loss)//this is not in the Dictionary
			{
				sprintf(FragName, "y%d-81", length - i - 1);
				CheckFragment(FragName, m_Mass0 - bMass - 63.998286 - NH3, likelihood* LESSCOMMON, 0, yIntensity, 0.005f);	//y ion
			}
		}

		if (NumU)
		{
			sprintf(FragName, "b%d-91", i + 1);
			if (m_sequence[i] == 'U')
				CheckFragment(FragName, bMass - 91.009184, likelihood, 1, bIntensity, 0.005f);	//common for b ion with C-term U
			else
				CheckFragment(FragName, bMass - 91.009184, likelihood*RARE, 0, bIntensity, 0.005f);	//uncommon (-91 is only common for b ions with C-term U)
		}
		if (NumU0 - NumU)
		{
			sprintf(FragName, "y%d-91", length - i - 1);
			CheckFragment(FragName, m_Mass0 - bMass - 91.009184, likelihood*RARE, 0, yIntensity, 0.005f);	//need to find out what is common for y ions
		}

		if (NumJ)
		{
			sprintf(FragName, "b%d-92", i + 1);
			if (m_sequence[i] == 'J')
				CheckFragment(FragName, bMass - 91.993200, likelihood, 1, bIntensity, 0.005f);	//common
			else
				CheckFragment(FragName, bMass - 91.993200, likelihood * RARE, 0, bIntensity, 0.005f);	//uncommon
		}
		if (NumJ0 - NumJ)
		{
			sprintf(FragName, "y%d-92", length - i - 1);
			CheckFragment(FragName, m_Mass0 - bMass - 91.993200, likelihood* RARE, 0, yIntensity, 0.005f);	//y ion
		}

		if (PhosY)
		{
			sprintf(FragName, "b%d-80", i + 1);
			CheckFragment(FragName, bMass - 79.966330, likelihood, 1, bIntensity, 0.005f);	//b ion
		}
		if (PhosY0 - PhosY)
		{
			sprintf(FragName, "y%d-80", length - i - 1);
			CheckFragment(FragName, m_Mass0 - bMass - 79.966330, likelihood, 1, yIntensity, 0.005f);	//y ion
		}

		if (PhosST + PhosY)
		{
			sprintf(FragName, "b%d-98", i + 1);
			CheckFragment(FragName, bMass - 97.976895, likelihood, 1, bIntensity, 0.005f);	//b ion
		}
		if (PhosST0 + PhosY0 - PhosST - PhosY)
		{
			sprintf(FragName, "y%d-98", length - i - 1);
			CheckFragment(FragName, m_Mass0 - bMass - 97.976895, likelihood, 1, yIntensity, 0.005f);	//y ion
		}

		if (length - i > 2)
		{
			sprintf(FragName, "y%d-Cterm", length - i - 1);
			CheckFragment(FragName, m_Mass0 - bMass - ResidueMass[length - 1], LESSCOMMON, 0, yIntensity, 0.005f);	//y ion - C-term residue
		}

		// c, x ions (positive offset)		
		sprintf(FragName, "b%d+17", i + 1);
		OffsetLikelihood = VERYRARE;
		if (m_sequence[i + 1] == 'Q')
			OffsetLikelihood = COMMON;

		if (i == 0)
			CheckFragment(FragName, bMass + NH3, likelihood * OffsetLikelihood, 1, bIntensity, 0.005f);	//c1 ion can be observed without b1
		else
			CheckFragment(FragName, bMass + NH3, likelihood * OffsetLikelihood, 0, bIntensity, 0.005f);	//c ion (b ion must be present)
		
		//x ion
		sprintf(FragName, "y%d+28", length - i - 1);
		CheckFragment(FragName, m_Mass0 - bMass + CO, likelihood * VERYRARE, 0, yIntensity, 0.005f);	//x is rare

		//other neutral losses
		//b ions
		for (j = 0; j < NUM_B_OFFSETS; j++)
		{
			if (bOffset[j] != 64 || !MetOx)
			{
				OffsetLikelihood = VERYRARE;
				if (bOffset[j] == 34 && NH3loss > 1)	//2NH3
					OffsetLikelihood = LESSCOMMON;
				else if (bOffset[j] == 35 && NH3loss)	//H2O + NH3 (note c-terminus of a b ion can lose H2O)
					OffsetLikelihood = LESSCOMMON;
				else if (bOffset[j] == 36 && H2Oloss)	//2H2O
					OffsetLikelihood = LESSCOMMON;
				else if (bOffset[j] == 44 && m_sequence[i] == 'T')
					OffsetLikelihood = LESSCOMMON;
				else if (bOffset[j] == 45 && NH3loss)				//a-NH3
					OffsetLikelihood = LESSCOMMON;
				else if (bOffset[j] == 46)	//a-H2O
					OffsetLikelihood = LESSCOMMON;
				else if (bOffset[j] == 51 && NH3loss > 2)	//b-3NH3
					OffsetLikelihood = RARE;
				else if (bOffset[j] == 52 && NH3loss > 1)	//b-2NH3 - H2O
					OffsetLikelihood = RARE;
				else if (bOffset[j] == 53 && H2Oloss && NH3loss)	//b-2H2O-NH3
					OffsetLikelihood = RARE;
				else if (bOffset[j] == 54 && H2Oloss > 1)	//b-3H2O
					OffsetLikelihood = RARE;
				else if (bOffset[j] == 55 && i == 0 && m_sequence[i] == 'W')
					OffsetLikelihood = LESSCOMMON;
				else if (bOffset[j] == 57 && i == 0 && m_sequence[i] == 'W')
					OffsetLikelihood = LESSCOMMON;
				else if (bOffset[j] == 62 && NH3loss > 1)	//a-2NH3
					OffsetLikelihood = RARE;
				else if (bOffset[j] == 63 && NH3loss)	//a-H2O-NH3
					OffsetLikelihood = LESSCOMMON;
				else if (bOffset[j] == 64 && H2Oloss)	//a-2H2O (note MetOx-64 has been consider earlier. this is only used when Met-Ox is not present)
					OffsetLikelihood = LESSCOMMON;

				if (bOffset[j] == 57 && (m_sequence[0] == 'G' || m_sequence[i] == 'G'))	//this is b(n-1)
					OffsetLikelihood = 0.0f;
				else if ((bOffset[j] == 42 || bOffset[j] == 59 || bOffset[j] == 60 || bOffset[j] == 61) && Arg == 0)	//b-3NH3
					OffsetLikelihood = 0.0;
				else if (bOffset[j] == 44 && Arg == 0 && m_sequence[i] != 'T')	//this is b(n-1)
					OffsetLikelihood = 0.0f;

				if (OffsetLikelihood > 0.0f)
				{
					sprintf(FragName, "b%d-%d", i + 1, bOffset[j]);
					CheckFragment(FragName, bMass - bOffset[j] * 1.0005, likelihood * OffsetLikelihood, 0, bIntensity, 0.05f);
				}
			}
		}

		//y ions
		for (j = 0; j < NUM_Y_OFFSETS; j++)
		{
			if (yOffset[j] != 64 || !(MetOx0 - MetOx))
			{
				OffsetLikelihood = VERYRARE;
				if (yOffset[j] == 34 && NH3loss0 - NH3loss > 1)	//y - 2NH3
					OffsetLikelihood = LESSCOMMON;
				else if (yOffset[j] == 35 && H2Oloss0 - H2Oloss > 0 && NH3loss0 - NH3loss > 0)			//y - NH3 - H2O
					OffsetLikelihood = LESSCOMMON;
				else if (yOffset[j] == 36 && H2Oloss0 - H2Oloss > 1)	//y - 2H2O
					OffsetLikelihood = LESSCOMMON;
				else if (yOffset[j] == 51 && NH3loss0 - NH3loss > 2)	//y - 3NH3
					OffsetLikelihood = RARE;
				else if (yOffset[j] == 52 && H2Oloss0 - H2Oloss > 0 && NH3loss0 - NH3loss > 1)	//y - H2O - 2NH3
					OffsetLikelihood = RARE;
				else if (yOffset[j] == 53 && H2Oloss0 - H2Oloss > 1 && NH3loss0 - NH3loss > 0)	//y - 2H2O - NH3
					OffsetLikelihood = RARE;
				else if (yOffset[j] == 54 && H2Oloss0 - H2Oloss > 2)	//y - 3H2O
					OffsetLikelihood = RARE;

				if (yOffset[j] == 57 && (m_sequence[i + 1] == 'G' || m_sequence[length - 1] == 'G'))	//this is y(n-1)
					OffsetLikelihood = 0.0f;
				else if (Arg0 - Arg == 0 && yOffset[j] == 42)// || yOffset[j] == 44 || yOffset[j] == 59 || yOffset[j] == 61))
					OffsetLikelihood = 0.0f;
				else if (yOffset[j] == 60 && m_sequence[length - 1] != 'R')	//C-term R
					OffsetLikelihood = 0.0f;

				if (OffsetLikelihood > 0.0f)
				{
					sprintf(FragName, "y%d-%d", length - i - 1, yOffset[j]);
					CheckFragment(FragName, m_Mass0 - bMass - yOffset[j] * 1.0005, likelihood * OffsetLikelihood, 0, yIntensity, 0.05f);
				}
			}
		}
	}

	//internal fragments (at least two residues)
	double fragmass;
	float likelihood1;
	for (i = 0; i < length - 3; i++)	//first cut (on the right side of i)
	{
		fragmass = 0.0;
		likelihood1 = GetCleavageLikelihood(i);
		MetOx = 0;
		H2Oloss = 0;
		NH3loss = 1;
		for (j = i + 1; j < length - 1; j++)	//second cut
		{
			fragmass += ResidueMass[j];
			if (m_sequence[j] == 'O')
				MetOx++;
			else if (m_sequence[j] == 'S' || m_sequence[j] == 'T' || m_sequence[j] == 'D' || m_sequence[j] == 'E')
				H2Oloss++;
			else if (m_sequence[j] == 'N' || m_sequence[j] == 'Q' || m_sequence[j] == 'R' || m_sequence[j] == 'K')
				NH3loss++;
			if (j > i + 1)	//at least two residues
			{
				for (k = 0; k < 8; k++)
				{
					bIntensity[k] = 0.0f;
				}

				bMaxCharge = (j - i) * 2 / 5;	//match criteria in CFeatureList for internal fragment
				if (bMaxCharge < 1)
					bMaxCharge = 1;
				else if (bMaxCharge > m_charge)
					bMaxCharge = m_charge;
				likelihood = sqrt(likelihood1 * GetCleavageLikelihood(j));
				sprintf(FragName, "[%d_%d]", i+2, j+1);
				CheckFragment(FragName, fragmass, likelihood*COMMON, 1, bIntensity, 0.005f);
				sprintf(FragName, "[%d_%d]-28", i + 2, j + 1);
				CheckFragment(FragName, fragmass - CO, likelihood * LESSCOMMON, 1, bIntensity, 0.005f);	//internal immonium
				if (NH3loss)
				{
					sprintf(FragName, "[%d_%d]-17", i + 2, j + 1);
					if (m_sequence[i+1] == 'Q')
						CheckFragment(FragName, fragmass - NH3, likelihood * COMMON, 0, bIntensity, 0.005f);
					else
						CheckFragment(FragName, fragmass - NH3, likelihood * RARE, 0, bIntensity, 0.005f);
				}
				if (H2Oloss)
				{
					sprintf(FragName, "[%d_%d]-18", i + 2, j + 1);
					if (m_sequence[i + 1] == 'E')
						CheckFragment(FragName, fragmass - H2O, likelihood * COMMON, 0, bIntensity, 0.005f);
					else
						CheckFragment(FragName, fragmass - H2O, likelihood * LESSCOMMON, 0, bIntensity, 0.005f);

					sprintf(FragName, "[%d_%d]-46", i + 2, j + 1);	//-CO-H2O
					if (m_sequence[i + 1] == 'E')
						CheckFragment(FragName, fragmass - CO - H2O, likelihood * LESSCOMMON, 0, bIntensity, 0.005f);
					else
						CheckFragment(FragName, fragmass - CO - H2O, likelihood* RARE, 0, bIntensity, 0.005f);	//internal immonium - H2O
				}

				if (MetOx)
				{
					sprintf(FragName, "[%d_%d]-64", i + 2, j + 1);
					CheckFragment(FragName, fragmass - 63.998286, likelihood*COMMON, 0, bIntensity, 0.005f);	//internal - 64
				}
			}
		}
	}

	//single amino acid
	int skip = 0, pyro = 0;
	char residue;
	for (i = 0; i < length; i++)
	{
		skip = 0;
		residue = m_sequence[i];
		if (m_sequence[i] == 'G' || m_sequence[i] == 'A')
			skip = 1;
		else
		{
			if (residue == 'I')
				residue = 'L';
			for (j = 0; j < i; j++)
			{
				if (residue == m_sequence[j] || (residue == 'L' && m_sequence[j] == 'I'))
				{
					skip = 1;
					break;
				}
			}
		}
		if (!skip)
		{
			for (k = 0; k < 8; k++)
			{
				bIntensity[k] = 0.0f;
				yIntensity[k] = 0.0f;
			}

			sprintf(FragName, "%c", residue);	//amino acid immonium ion
			CheckFragment(FragName, ResidueMass[i] - CO, VERYCOMMON, 1, bIntensity, 0.005f);
			sprintf(FragName, "%c'", residue);	//amino acid acyl
			if (residue == 'K')
				CheckFragment(FragName, ResidueMass[i], VERYCOMMON, 1, bIntensity, 0.005f);	//K' is very common
			else
				CheckFragment(FragName, ResidueMass[i], VERYRARE, 0, bIntensity, 0.005f);

			//special single-residue ion K-17 and pyroE
			if (m_sequence[i] == 'K')
			{
				sprintf(FragName, "%c-17", m_sequence[i]);	//K - NH3
				CheckFragment(FragName, 83.0735, VERYCOMMON, 1, bIntensity, 0.005f);
			}
			if (!pyro && (m_sequence[i] == 'E' || m_sequence[i] == 'Q'))
			{
				sprintf(FragName, "pyroE");	//Q-NH3 or E-H2O
				CheckFragment(FragName, 83.0371, VERYCOMMON, 1, bIntensity, 0.005f);
				pyro = 1;
			}
		}
	}
	delete[] ResidueMass;
}


//this function is used for both annotation and spectrum generation
//return detected charge prime number product (1 if not detected)
//When Common==1, assign byIntensity[z] values for different charge z. when Common==0, use byIntensity[] to score the ID (penalty applied for uncommon fragments)
int CMS2Spectrum::CheckFragment(char FragName[], double FragMass, float likelihood, int Common, float byIntensity[], float MinTolerance)
{
	//set charge according to mass instead of residue count (residue count will favor longer fragment for higher charge state) - ignore the passed parameter
	//note: some annotations are never exported to .ann file because they are not in the Dictionary
	int MaxCharge = (int)(FragMass / 200.0);	//2+ for 400 Da
	if (MaxCharge > m_charge)
		MaxCharge = m_charge;
	if (MaxCharge < 1)
		MaxCharge = 1;

	int z1 = 1;
	int z2 = MaxCharge;

	if (strstr(FragName, "Mol") == FragName)
	{
		z1 = z2 = m_charge;
	}
			
	int z;
	float accuracy, accuracy0 = (float)FragMass*1.5e-5f;	//15 ppm. accuracy and accuracy0 is for monosiotopic peak vs theoretical. 0 stands for zero charge
	if (accuracy0 < MinTolerance)
		accuracy0 = MinTolerance;
	float accuracy_isotope, accuracy_isotope0 = accuracy0;	//accuracy_isotope is for heavy isotopic peak vs theoretical calculated from top isotope m/z, and therefore more accurate
	if (MinTolerance > 0.02f)
	{
		accuracy_isotope0 = (float)FragMass * 1.5e-5f;	//15 ppm;
		if (accuracy_isotope0 < 0.003f)
			accuracy_isotope0 = 0.003f;
	}
	
	int i;
	char NewLabel[20];

	if (m_Is4Annotation)
	{
		//calculate isotope distribution
		float a[64], a2[64];	//isotope distribution. a[1] is monoisotopic
		int index[64];		//index of spectrum for each isotope in a2[]
		int i, j, top, top_index;
		
		int n = IsotopeSimulationFromMass(FragMass, (int)(7.0 + FragMass / 1000.0), a, 0.1f, &top);	//top is the index of top of a[]

		float max = a[top];
		double mz, topmz;
		float currentmz;
		int topindex_real, index1_real, index2_real, top2;	//indices for real spectrum and a[]
		float sum1, sumxy1, sum2, sumxy2, product, simscore, score;
		float FeatureIntensity, MinSimscore;

		for (z = z1; z <= z2; z++)
		{
			if (Common || byIntensity[z] > 0.0f)
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

				index[top] = top_index;
				j = top + 1;	//index in a[]
				index2_real = index[top];	//index of the last isotope
				while (j < n && i < m_points)
				{
					currentmz = (float)(topmz + (j - top) * DELTA_ISOTOPE / z);
					while (i < m_points && m_mass[i] < currentmz - accuracy_isotope)
					{
				//		index2_real = i;	//index of the last isotope
						i++;
					}
					while (i < m_points && m_mass[i] < currentmz + accuracy_isotope)
					{
						if (a2[j] < m_intensity[i])
						{
							a2[j] = m_intensity[i];
							index[j] = i;
						}
						index2_real = i;	//index of the last isotope
						i++;
					}
					if (index[j] >= 0)
					{
						if (a2[j] > a2[top])
						{
							top2 = j;	//index for peak top in a2[]
							topindex_real = index[j];
						}
					}
					else
						break;
					j++;
				}

				i = top_index;
				j = top - 1;
				index1_real = index[top];	//index of the first isotope
				while (j >= 0 && i >= 0)
				{
					currentmz = (float)(topmz - (top - j) * DELTA_ISOTOPE / z);
					while (i >= 0 && m_mass[i] > currentmz + accuracy_isotope)
					{
			//			index1_real = i;	//index of the first isotope
						i--;
					}
					while (i >= 0 && m_mass[i] > currentmz - accuracy_isotope)
					{
						if (a2[j] < m_intensity[i])
						{
							a2[j] = m_intensity[i];
							index[j] = i;
						}
						index1_real = i;	//index of the first isotope
						i--;
					}
					if (index[j] >= 0)
					{
						if (a2[j] > a2[top2])
						{
							top2 = j;
							topindex_real = index[j];
						}
					}
					else
						break;
					j--;
				}

				//test whether a[] and a2[] are too different before calculating similarity
				if (top2 != top)
				{
					if (a[top2] < a[top] * 0.4f || a2[top] < a2[top2]*0.4f)
						continue;
					else if (abs(top2 - top) > n / 2 || abs(top2 - top) > (int)sqrt(FragMass) / 16 - 1)
						continue;
				}

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
				if (index1_real >= 0 && index2_real > index1_real)
				{
					sum2 = 0.0f;
					for (i = index1_real; i <= index2_real; i++)
						sum2 += m_intensity[i];
				}

				simscore = product / (float)sqrt((double)(sum1 * sum2));
				score = simscore * likelihood;
				
				//for uncommon fragment, reduce score based on its parent intensity (reduce to no less than half of original score)
				if (!Common)
					score *= (1.0f + 0.1f * log2(byIntensity[z] / m_MaxSignal));	//decrease by 0.1 for every 2x decrease - 0.8 for 25% of parent b/y signal

				//store labeling result
				MinSimscore = 0.5f;
				if (fabs(topmz - m_PrecursorMz) < 1.9)	//in the isolation window (interferring peaks are often co-isolated)
					MinSimscore = 0.85f;				

				if (simscore > MinSimscore && score > 0.3f && score > m_labelscore[topindex_real] * 0.8f)	//isotope similarity (simscore) at least 0.5
				{
					if (Common)
						byIntensity[z] += FeatureIntensity;

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
							if (index[i] != topindex_real)
								m_label[index[i]] = "";
						}
					}
				}
			}
		}
	}
	else
	{
		//spectrum generation
//		float x_env[32], y_env[32]; //one isotope envelop to be added
		//Generate isotope envelop and then add to spectrum
		for (z = z2; z >= z1; z--)	//higher charge first to be faster
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
					if (m_FeatureIntensity[i] > 0.0f && FragMass/z + PROTON > m_StartMass && FragMass / z + PROTON < m_EndMass)
					{
						m_FragIonMz[m_NumFragIons] = (float)(FragMass / z + PROTON);
						m_FragIonCharge[m_NumFragIons] = z;
						m_FragIonIntensity[m_NumFragIons] = m_FeatureIntensity[i];
						m_NumFragIons++;
					}
					break;
				}
			}
		}
	}
	return 1;
}


//return the likelihood of this cleavage
float CMS2Spectrum::GetCleavageLikelihood(int CleavageSite)
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

//quickly estimate isotope distribution of peptide from mass
//a[1] is monoisotopic, a[0] = 0. 
int CMS2Spectrum::IsotopeSimulationFromMass(double mass, int size, float a[], float cutoff, int *top)
{
	double p[15] = {3.3138,1.6648,2.0832,-17.077,1.168,-1.0828,2.1053,-0.9349,0.76608,3.9292,-0.3259,-9.3486,0.19216,0.06216,5.4696};
	double MassE4 = 1.0e-4 * mass;	//mass is less than 10,000 Da, so MassE4 < 1.0

	double m1 = p[12]/sqrt(MassE4);
	double m2 = p[13]/MassE4;
	double m3 = p[14]*MassE4;

	//gaussian shape if > 10kDa
	int n;
	a[0] = 0.0f;	//a[1] is monoisotopic
	for (n = 0; n < size - 1; n++)
		a[n + 1] = (float)(m1 * exp(-m2 * (n - m3) * (n - m3)));

	if (MassE4 < 1.0)
	{
		m1 = p[0] * exp(p[1] * MassE4);
		m2 = p[2] * exp(p[3] * MassE4) + p[4] + p[5] * MassE4;
		m3 = p[6] * exp(p[7] * MassE4);
		double m4 = p[8] + p[9] * MassE4 + p[10] * exp(p[11] * MassE4);
		for (n = 0; n < size - 1; n++)
			a[n + 1] = (float)(MassE4 * a[n + 1] + (1.0 - MassE4) * exp(-m1 / (n + m3) - m2 * (n - m4)));
	}
	

	//truncate and normalize
	float max = 0.0f, sum = 0.0f;
	n = 1;
	*top = 1;
	while (n < size && (n == 1 || a[n] >= cutoff * max))
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
	for (i = 1; i < n; i++)
		a[i] /= sum;

	if (*top > 1 && a[*top - 1] > a[*top] * 0.9f)
		*top = *top - 1;
	return n;
}

//export features for this spectrum
void CMS2Spectrum::ExportAnnotations(char filename[])
{
	using namespace std;
	ofstream file(filename, ios::app);	//append to file

	//sequence
	file << '>' << "length=" << m_sequence.length() << "; sequence=" << m_sequence << "; charge=" << m_charge << "; NCE=" << (int)(m_CollisionEnergy + 0.5f) << "; mass range=[" << m_StartMass << '-' << m_EndMass << ']' << '\n';

	//annotations
	int j, k, iStart, b, success;
	char OneLabel[20];

	for (j = 0; j < m_points; j++)
	{
		if (m_label[j].length() > 0)
		{
			iStart = 0;
			if ((b = (int)m_label[j].find('\n', iStart)) > 0)	//Export only the top label. Change "if" to "while" if export other possible labels for this m/z (causing problem due to redundent intensity prediction)
			{
				//a label is from iStart to b
				m_label[j].copy(OneLabel, b - iStart, iStart);
				OneLabel[b - iStart] = '\0';

				if (OneLabel[0] == 'I')	//I or I' is not in the dictionary
					OneLabel[0] = 'L';

				//search OneLabel in the feature list (Dictionary)
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
//			file << k << ',' << m_FeatureList->m_Features[k] << ',' << m_FeatureIntensity[k] << '\n';
			file << k << ',' << m_FeatureIntensity[k] << '\n';
	}
	file.close();
}


//export features for this spectrum
//not used right now
void CMS2Spectrum::ExportSpectrum(char filename[])
{
	using namespace std;
	ofstream file(filename, ios::app);	//append to file

	file << "Name: " << m_sequence << '/' << m_charge << '\n';
	file << "Comment: Charge=" << m_charge << " NCE=" << (int)(m_CollisionEnergy + 0.5f) << '\n';
	file << "Num peaks: " << m_points << '\n';

	int i;
	for (i = 0; i < m_points; i++)
		file << m_mass[i] << '\t' << m_intensity[i] << '\n';
	file << '\n';
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

	//new spectrum has smaller m/z	(not possible as currently implemented)
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
	int p1 = 0;
	if (x2[0] > m_mass[0])
	{
		//because inserted ions are sorted by monoisotopic m/z, the insertion point must be very close to the end
		p1 = m_points - 1;
		while (m_mass[p1] > x2[0])
			p1--;
	}

	int start = p1;
	int p2 = 0;

	int index = 0;
	while (p1 < m_points && p2 < n2)
	{
		if (x2[p2] - m_mass[p1] > tol)
		{
			x[index] = m_mass[p1];
			y[index] = m_intensity[p1];
			index++;
			p1++;
		}
		else if (m_mass[p1] - x2[p2] > tol)
		{
			x[index] = x2[p2];
			y[index] = y2[p2];
			index++;
			p2++;
		}
		else
		{
			while (p1 + 1 < m_points && fabs(m_mass[p1 + 1] - x2[p2]) < fabs(m_mass[p1] - x2[p2]))
			{
				x[index] = m_mass[p1];
				y[index] = m_intensity[p1];
				index++;
				p1++;
			}

			if (m_intensity[p1] + y2[p2] > 0.0f)
				x[index] = (m_mass[p1] * m_intensity[p1] + x2[p2] * y2[p2]) / (m_intensity[p1] + y2[p2]);
			else
				x[index] = m_mass[p1];
			y[index] = m_intensity[p1] + y2[p2];
			index++;
			p1++;
			p2++;
		}
	}

	if (p1 == m_points)
	{
		memcpy(x + index, x2 + p2, (n2 - p2) * sizeof(float));
		memcpy(y + index, y2 + p2, (n2 - p2) * sizeof(float));
		index += (n2 - p2);
	}
	else if (p2 == n2)
	{
		memcpy(x + index, m_mass + p1, (m_points - p1) * sizeof(float));
		memcpy(y + index, m_intensity + p1, (m_points - p1) * sizeof(float));
		index += (m_points - p1);
	}

	//copy x[] to m_mass[] from start
	memcpy(m_mass + start, x, index * sizeof(float));
	memcpy(m_intensity + start, y, index * sizeof(float));
	m_points = start + index;
	return m_points;
}

//using golden-fraction method to search the index of mass0
//initialize from and to as the first and last points
//not used anymore
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

/*
//generate the spectrum from fragment ion information stored in m_FragIonMz[] etc.
void CMS2Spectrum::GenerateSpectrum()
{
	m_points = 0;

	//first sort the list based on m/z
	int i;
	for (i = 0; i < m_NumFragIons; i++)
		m_SortIndex[i] = i;

	using namespace std;
	float* arr = m_FragIonMz;
	std::sort(m_SortIndex, m_SortIndex + m_NumFragIons, [&arr](int i, int j) {return arr[i] < arr[j]; });

	//then add each fragment ion into the spectrum
	int j, k, n, top;
	float FragMass;
	float a[64];	//isotope distribution. a[1] is monoisotopic
	float x_env[64], y_env[64];	//isotope envelop to add into spectrum
	for (i = 0; i < m_NumFragIons; i++)
	{
		k = m_SortIndex[i];
		FragMass = (m_FragIonMz[k] - (float)PROTON) * m_FragIonCharge[k];
		n = IsotopeSimulationFromMass(FragMass, (int)(7.0 + FragMass / 1000.0), a, 0.2f, &top);	//top is the index of top of a[] (cutoff good 0.2 or 0.25)

		for (j = 0; j < n - 1; j++)
		{
			x_env[j] = m_FragIonMz[k] + j * (float)DELTA_ISOTOPE / m_FragIonCharge[k];
			y_env[j] = a[j + 1] * m_FragIonIntensity[k];	//a[1] is monoisotopic
		}
		AddToSpectrum(n - 1, x_env, y_env);
	}
}
*/


//use combin[][] to calculate (does not seem to work better)
void CMS2Spectrum::GenerateSpectrum()
{
	m_points = 0;

	//first sort the list based on m/z
	int i;
	for (i = 0; i < m_NumFragIons; i++)
		m_SortIndex[i] = i;

	using namespace std;
	float* arr = m_FragIonMz;
	std::sort(m_SortIndex, m_SortIndex + m_NumFragIons, [&arr](int i, int j) {return arr[i] < arr[j]; });

	//then add each fragment ion into the spectrum
	int j, k, m, n, top;
	float FragMass;
	float a[64], Parent[64];	//isotope distribution. a[1] is monoisotopic
	float x_env[64], y_env[64];	//isotope envelop to add into spectrum
	double x;
	n = IsotopeSimulationFromMass(m_Mass0, (int)(7.0 + m_Mass0 / 1000.0), Parent, 0.05f, &top);	//top is the index of top of Parent[]. parent[1] is monoisotopic
	
	//apply isolation window
	float IsolationWidth1 = 0.7f;	//half of isolation window (optimized: 0.7-0.95 ok, 0.7 best)
	float IsolationWidth2 = 0.7f;
	float sum = 0.0f;
	for (j = 1; j < n; j++)
	{
		if (top - j > IsolationWidth1 * m_charge || j - top > IsolationWidth2 * m_charge)
			Parent[j] = 0.0f;
		else
			sum += Parent[j];

	}
	while (n > 1 && Parent[n - 1] == 0.0f)
		n--;
	n--;	//because Parent[0] is nothing
	if (n > 16)
		n = 16;	//because combin[16][16]

	//normalize
	for (i = 1; i <= n; i++)
		Parent[i] /= sum;

	for (i = 0; i < m_NumFragIons; i++)
	{
		k = m_SortIndex[i];
		FragMass = (m_FragIonMz[k] - (float)PROTON) * m_FragIonCharge[k];

		if (FragMass > m_Mass0 - 36.1)
			memcpy(a, Parent+1, n * sizeof(float));
		else
		{
			x = FragMass / m_Mass0;
			for (j = 0; j < n; j++)
			{
				a[j] = 0.0f;	//a[0] is monoisotopic
				for (m = j; m < n; m++)
					a[j] += Parent[m+1] * combin[j][m] * (float)pow(1.0 - x, m - j);
				a[j] *= (float)pow(x, j);
			}
		}

		for (j = 0; j < n; j++)
		{
			x_env[j] = m_FragIonMz[k] + j * (float)DELTA_ISOTOPE / m_FragIonCharge[k];
			y_env[j] = a[j] * m_FragIonIntensity[k];
		}
		AddToSpectrum(n, x_env, y_env);
	}
}
