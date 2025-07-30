// MS2FeatureConverter.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include "CMS2Spectrum.h"

void trim_newline(char* str) {
    int len = strlen(str);
    while (len > 0 && (str[len-1] == '\n' || str[len-1] == '\r')) {
        str[--len] = '\0';
    }
}

int main(int argc, char* argv[])
{
    // Remove the hardcoded argc = 2 to allow proper CLI usage
    if (argc < 2)
    {
        std::cout << "Please specify file name\n";
        return 1;
    }

    char* inputfilename = argv[1];
    //char inputfilename[1024] = "C:\\Users\\zhang\\Documents\\MSData\\IARPA3_best_tissue_add_info.msp.txt";
    char outputfilename[1024];
    if (argc > 2)
        strcpy(outputfilename, argv[2]);
    else
    {
        strcpy(outputfilename, inputfilename);
        strcat(outputfilename, "_ann.txt");
    }

    if (strstr(inputfilename, ".msp") != NULL)  //NIST .msp file
    {
        using namespace std;

        ifstream file(inputfilename, ios::in);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open the file!" << std::endl;
            return 1;
        }

        //generate feature list
        CFeatureList *FeatureList = new CFeatureList();
        char Dictionaryfilename[1024];
        strcpy(Dictionaryfilename, inputfilename);
        char *temp = strrchr(Dictionaryfilename, '/');  // Use forward slash for macOS
        if (temp != NULL) {
            strcpy(temp, "/Dictionary.txt");  // Use forward slash for macOS
        } else {
            // If no path separator found, just use current directory
            strcpy(Dictionaryfilename, "Dictionary.txt");
        }
        FeatureList->SaveAs(Dictionaryfilename);

        //generate output file
        ofstream file_out(outputfilename, ios::out);
        file_out << "Dictionary size = " << FeatureList->m_NumFeatures << '\n';
        file_out.close();

        char charline[1024];
        file.getline(charline, 1023, '\n');

        int b;
        int length = (int)strlen(charline);
        char sequence[64] = "";

        int mod_count, index;
        char* c;
        char residue, mod[128];
        int keep = 1, charge = 0;
        long i, NumIons;
        float PrecursorMz=0.0f, CollisionEnergy=0.0f;
        float* mass, * intensity;
        double Mass0 = 0.0;   //peptide mass
        while (!(length == 0 && strlen(sequence) == 0))
        {
            if (strstr(charline, "Name:") == charline)
            {
                cout << "charline: " << charline << endl;
                charge = 0;
                Mass0 = 0.0;
                PrecursorMz = 0.0f;
                CollisionEnergy = 0.0f;

                b = 6;
                while (charline[b] != '/')
                {
                    sequence[b - 6] = charline[b];
                    b++;
                }
                sequence[b-6] = '\0';

                cout << "sequence: " << sequence << endl;

                //read modification from parenthesis (only takes JUOsty)
                mod_count = 0;
                c = strchr(charline, '(');
                while (c != NULL)
                {
                    sscanf(c, "(%d,%c,%s", &index, &residue, mod);
                    c = strchr(mod, ')');
                    if (c != NULL) {
                        *c = '\0';
                    }
                    cout << "index: " << index << ", residue: " << residue << ", mod: " << mod << endl;
                    if (sequence[index] == residue)
                    {
                        if (sequence[index] == 'C' && strcmp(mod, "CAM") == 0)
                            sequence[index] = 'U';
                        else if (sequence[index] == 'C' && strcmp(mod, "CM") == 0)
                            sequence[index] = 'J';
                        else if (sequence[index] == 'M' && strcmp(mod, "Oxidation") == 0)
                        {
                            sequence[index] = 'O';
                            mod_count++;
                        }
                        else if (sequence[index] == 'S' && strcmp(mod, "Phosphorylation") == 0)
                        {
                            sequence[index] = 's';
                            mod_count++;
                            keep = 0;   //do not consider phosphorylation for now
                        }
                        else if (sequence[index] == 'T' && strcmp(mod, "Phosphorylation") == 0)
                        {
                            sequence[index] = 't';
                            mod_count++;
                            keep = 0;
                        }
                        else if (sequence[index] == 'Y' && strcmp(mod, "Phosphorylation") == 0)
                        {
                            sequence[index] = 'y';
                            mod_count++;
                            keep = 0;
                        }
                        else {
                            cout << "Skipping spectrum with mod_count " << mod_count << " and keep " << keep << endl;
                            keep = 0;
                        }
                    }
                    else
                        keep = 0;

                    c = strchr(c + 1, '(');
                }
                if (mod_count > 1)
                    keep = 0;
            }
            else if (strstr(charline, "Comment:") == charline)
            {
                if ((c = strstr(charline, "Charge=")) != NULL && strlen(sequence) > 0)
                {
                    charge = atoi(c + 7);
                }
                if ((c = strstr(charline, "Parent=")) != NULL && strlen(sequence) > 0)
                {
                    PrecursorMz = (float)atof(c + 7);
                }
                if ((c = strstr(charline, "NCE=")) != NULL && strlen(sequence) > 0)
                {
                    CollisionEnergy = (float)atof(c + 4);
                }
            }
            else if ((c = strstr(charline, "Num peaks:")) == charline && length > 11)
            {
                NumIons = (int)atoi(c+10);
                if (NumIons > 0)
                {
                    mass = new float[NumIons];
                    intensity = new float[NumIons];

                    for (i = 0; i < NumIons; i++)
                    {
                        file.getline(charline, 512, '\n');
                        trim_newline(charline);
                        sscanf(charline, "%f\t%f", &mass[i], &intensity[i]);  // Use standard sscanf instead of sscanf_s
                    }

                    if (charge != 0 && keep)
                    {
                        //send spectrum to CMS2Spectrum class
                        if (strlen(sequence) <= FeatureList->m_MaxNumResidues)
                        {
                            CMS2Spectrum spectrum(NumIons, mass, intensity, sequence, charge, CollisionEnergy, PrecursorMz);
                            spectrum.Annotate();
                            spectrum.ExportAnnotations(outputfilename, FeatureList);
                        }
                    }
                }
            }
            else if (length == 0)   //blank line, finish entry
            {
                //initialize next entry
                sequence[0] = '\0';
                keep = 1;
                CollisionEnergy = 30.0f;
                charge = 0;
                PrecursorMz = 0.0f;
                Mass0 = 0.0;
                NumIons = 0;
            }
            file.getline(charline, 1023, '\n');
            trim_newline(charline);
            length = (int)strlen(charline);
        }
        delete FeatureList;
        file.close();
    }
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
