// SpecSimilarity.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include "CSpectrum.h"

//read next spectrum and return ion count
int ReadNextSpectrum(std::ifstream& file, float* mass, float* intensity, char sequence[], int* charge, float* CollisionEnergy)
{
    int b, i;
    char* c, * c1;

    int NumIons = 0;
    int mod_count, keep = 1, index;
    char charline[1024], residue, mod[256];
    file.getline(charline, 1023, '\n');
    int length = (int)strlen(charline);
    sequence[0] = '\0';
    while (!(length == 0 && strlen(sequence) == 0))
    {
        if (strstr(charline, "Name:") == charline)
        {
            *charge = 0;
            *CollisionEnergy = 0.0f;

            b = 6;
            while (charline[b] != '/')
            {
                sequence[b - 6] = charline[b];
                b++;
            }
            sequence[b - 6] = '\0';
            sscanf(charline + b, "/%d", charge);

            //read modification from parenthesis (only takes JUOsty)
            mod_count = 0;
            c = strchr(charline, '(');
            while (c != NULL && keep)
            {
                sscanf(c, "(%d,%c,%s", &index, &residue, mod);
                if ((c1 = strchr(mod, ')')) != NULL)
                    *c1 = '\0';
                if (sequence[index] == residue)
                {
                    if (sequence[index] == 'C' && (strcmp(mod, "CAM") == 0 || strcmp(mod, "Carbamidomethyl") == 0))
                        sequence[index] = 'U';
                    else if (sequence[index] == 'C' && (strcmp(mod, "CM") == 0 || strcmp(mod, "Carboxymethyl") == 0))
                        sequence[index] = 'J';
                    else if (sequence[index] == 'M' && strcmp(mod, "Oxidation") == 0)
                    {
                        sequence[index] = 'O';
                        mod_count++;
                    }
                    else if (sequence[index] == 'S' && strcmp(mod, "Phospho") == 0)
                    {
                        sequence[index] = 's';
                        mod_count++;
                        keep = 0;   //do not consider phosphorylation for now
                    }
                    else if (sequence[index] == 'T' && strcmp(mod, "Phospho") == 0)
                    {
                        sequence[index] = 't';
                        mod_count++;
                        keep = 0;
                    }
                    else if (sequence[index] == 'Y' && strcmp(mod, "Phospho") == 0)
                    {
                        sequence[index] = 'y';
                        mod_count++;
                        keep = 0;
                    }
                    else
                        keep = 0;
                }
                else
                    keep = 0;

                c = strchr(c + 1, '(');
            }


            if ((c = strstr(charline, "_NCE")) != NULL && strlen(sequence) > 0)
            {
                *CollisionEnergy = (float)atof(c + 4);
            }
        }
        else if (strstr(charline, "Comment:") == charline)
        {
            if ((c = strstr(charline, "Charge=")) != NULL && strlen(sequence) > 0)
            {
                *charge = atoi(c + 7);
            }
            if ((c = strstr(charline, "NCE=")) != NULL && strlen(sequence) > 0)
            {
                *CollisionEnergy = (float)atof(c + 4);
            }          
        }
        else if ((c = strstr(charline, "Num peaks:")) == charline && length > 11)
        {
            NumIons = (int)atoi(c + 10);
            if (NumIons > 0)
            {
                for (i = 0; i < NumIons; i++)
                {
                    file.getline(charline, 512, '\n');
                    sscanf(charline, "%f\t%f", &mass[i], &intensity[i]);
                }
            }
        }
        else if (length == 0)   //blank line, finish entry
            return NumIons;

        file.getline(charline, 1023, '\n');
        length = (int)strlen(charline);
    }

    if (keep)
        return NumIons;
    else
        return ReadNextSpectrum(file, mass, intensity, sequence, charge, CollisionEnergy);
}

/*
This program compares two datasets containing the experimental spectra and predicted spectra, and save similarity scores into a .sim file
inputfilename1 contains the predicted spectra, and inputfilename2 contains the experimental spectra (.ms2)
*/

int main()
{
    using namespace std;

    char inputfilename1[1024] = "C:\\Users\\zhang\\Documents\\MSData\\UniSpec_Datasets3\\Predicted\\FullMS2FormerPredict_AItrain_QEHumanCho_2022418v2.msp.txt";    //the first file contains the predicted spectra
    char inputfilename2[1024] = "C:\\Users\\zhang\\Documents\\MSData\\UniSpec_Datasets3\\Predicted\\AItrain_QEHumanCho_2022418v2.ms2.txt";   //the second file is experimental ms2 file

    //program exports two output files: 1) .sim file, containing similarity score of every spectrum; 2) .dis file, containing the similarity distribution profile of the dataset
    char outputfilename[1024], outputfilename2[1024];

    ifstream file1(inputfilename1, ios::in);
    ifstream file2(inputfilename2, ios::in);
    if (!file1.is_open() || !file2.is_open())
    {
        std::cerr << "Error: Could not open the file!" << std::endl;
        return 1;
    }

    strcpy(outputfilename, inputfilename1); //set output file name as .sim
    char *c = strstr(outputfilename, ".msp");
    if (c == NULL)
        c = strstr(outputfilename, ".ms2");
    c[1] = 's';
    c[2] = 'i';
    c[3] = 'm';

    strcpy(outputfilename2, inputfilename1); //set output file name as .dis
    c = strstr(outputfilename2, ".msp");
    if (c == NULL)
        c = strstr(outputfilename2, ".ms2");
    c[1] = 'd';
    c[2] = 'i';
    c[3] = 's';

    ofstream fileout(outputfilename, ios::out);
    fileout << "sequence,charge,NCE,similarity\n";

    ofstream fileout2(outputfilename2, ios::out);
    fileout2 << "similarity,count\n";

    //read spectra into mass1, instensity1, etc.
    int n1 = 1, n2 = 1;
    float similarity;
    float* mass1 = new float[131072];
    float* intensity1 = new float[131072];
    float* mass2 = new float[131072];
    float* intensity2 = new float[131072];

    char sequence1[64], sequence2[64];
    int charge1, charge2;
    float CollisionEnergy1, CollisionEnergy2;
    float avg = 0.0f;
    int count = 0;
    int ScoreDistribution[101];
    memset(ScoreDistribution, 0, 101 * sizeof(int));
    
    while (n1 && n2)
    {
        n1 = ReadNextSpectrum(file1, mass1, intensity1, sequence1, &charge1, &CollisionEnergy1);
        n2 = ReadNextSpectrum(file2, mass2, intensity2, sequence2, &charge2, &CollisionEnergy2);

        if (n1 && n2)
        {
            //read next spectrum1 if not the same
            while (n1 && (strcmp(sequence1, sequence2) != 0 || charge1 != charge2 || CollisionEnergy1 != CollisionEnergy2))
                n1 = ReadNextSpectrum(file1, mass1, intensity1, sequence1, &charge1, &CollisionEnergy1);
        }

        if (n1 && n2 && strcmp(sequence1, sequence2) == 0 && charge1 == charge2 && CollisionEnergy1 == CollisionEnergy2)
        {
            CSpectrum spectrum(n1, mass1, intensity1);
            similarity = spectrum.Similarity(n2, mass2, intensity2, 0.05f);

            //save similarity value to outputfile
            fileout << sequence1 << ',' << charge1 << ',' << CollisionEnergy1 << ',' << similarity << '\n';
            avg += similarity;
            count++;

            ScoreDistribution[(int)(similarity * 100.0f +  0.5f)]++;
        }
    }
    avg /= count;

    int i;
    for (i = 0; i <= 100; i++)
        fileout2 << i * 0.01 << '\t' << ScoreDistribution[i] << '\n';

    fileout2 << "\naverage similarity = " << avg << "(n = " << count << ")\n";

    delete[] mass1;
    delete[] intensity1;
    delete[] mass2;
    delete[] intensity2;

    file1.close();
    file2.close();
    fileout.close();
    fileout2.close();
}


