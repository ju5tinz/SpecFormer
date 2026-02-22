#pragma once
#include <string>

#define MAXCHARGE	5	//Max charge of a fragment ion
#define MAXLENGTH	40
#define NUM_B_OFFSETS	17
#define NUM_Y_OFFSETS	12
#define OFFSETSIZE	65

class CFeatureList
{
public:
	CFeatureList();
	~CFeatureList();
	void SaveAs(char[]);

	//Feature list
	int m_MaxNumResidues;
	int m_MaxCharge;
	int m_NumFeatures;
	std::string* m_Features;
};

