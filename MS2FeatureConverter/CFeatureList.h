#pragma once
#include <string>

#define OFFSET_FROM 30
#define OFFSET_TO 64

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

