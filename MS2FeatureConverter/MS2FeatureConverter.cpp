// MS2FeatureConverter.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


#include "CMS2Spectrum.h"

int main(int argc, char* argv[])
{
    argc = 2;
    if (argc < 2)
    {
        std::cout << "Please specify file name\n";
        return 1;
    }

    using namespace std;

//    char* inputfilename = argv[1];
//    char inputfilename[1024] = "C:\\Users\\zhang\\Documents\\MSData\\IARPA3_best_tissue_add_info.msp.txt";
 //   char inputfilename[1024] = "C:\\Users\\zhang\\Documents\\MSData\\IARPA3_best_tissue_add_info.ann.txt";
    char inputfilename[1024] = "C:\\Users\\zhang\\Documents\\MSData\\output_07_31_25.ann.txt";

    char outputfilename[1024];
    char* b;
    if (strstr(inputfilename, ".msp") != NULL)  //NIST .msp file
    {
        if (argc > 2)
            strcpy(outputfilename, argv[2]);
        else
        {
            strcpy(outputfilename, inputfilename);
            b = strstr(outputfilename, ".msp");
            b[1] = 'a';
            b[2] = 'n';
            b[3] = 'n';
        }

        ifstream file(inputfilename, ios::in);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open the file!" << std::endl;
            return 1;
        }

        //generate feature list
        CFeatureList *FeatureList = new CFeatureList();
        float* FeatureIntensity = new float[FeatureList->m_NumFeatures];
//        int* Count = new int[FeatureList->m_NumFeatures];
//        memset(Count, 0, FeatureList->m_NumFeatures * sizeof(int));
        char Dictionaryfilename[1024];
        strcpy(Dictionaryfilename, inputfilename);
        char *temp = strrchr(Dictionaryfilename, '\\');
        strcpy(temp, "\\Dictionary.txt");
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


//        int NLcount[64], fraglen, neutral;
//        char fragname;
//        memset(NLcount, 0, 64 * sizeof(int));
        while (!(length == 0 && strlen(sequence) == 0))
        {
            if (strstr(charline, "Name:") == charline)
            {
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

                //read modification from parenthesis (only takes JUOsty)
                mod_count = 0;
                c = strchr(charline, '(');
                while (c != NULL)
                {
                    sscanf(c, "(%d,%c,%s", &index, &residue, mod);
                    c = strchr(mod, ')');
                    *c = '\0';
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
                        else
                            keep = 0;
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
                        sscanf(charline, "%f\t%f", &mass[i], &intensity[i]);
                    }

                    if (charge != 0 && keep)
                    {
                        //send spectrum to CMS2Spectrum class
                        if (strlen(sequence) <= FeatureList->m_MaxNumResidues)
                        {
                            for (i = 0; i < FeatureList->m_NumFeatures; i++)
                                FeatureIntensity[i] = 0.0f;
                            CMS2Spectrum spectrum(NumIons, mass, intensity, sequence, charge, CollisionEnergy, PrecursorMz, FeatureList, FeatureIntensity);
                            spectrum.Annotate();
                            spectrum.ExportAnnotations(outputfilename);
               /*           for (i = 0; i < FeatureList->m_NumFeatures; i++)
                            {
                                if (FeatureIntensity[i] > 0.5f)
                                {
                                    Count[i]++;
                                    if (sscanf(FeatureList->m_Features[i].c_str(), "%c%d-%d", &fragname, &fraglen, &neutral) == 3 && neutral >= OFFSET_FROM && neutral <= OFFSET_TO)
                                        NLcount[neutral]++;
                                }
                            }
                */
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
            length = (int)strlen(charline);
        }

/*
        for (i = OFFSET_FROM; i <= OFFSET_TO; i++)
        {
            if (!NLcount[i])
                cout << i << '\n';
        }
*/

        delete [] FeatureIntensity;
        delete FeatureList;
//        delete[] Count;
        file.close();
    }
    else if (strstr(inputfilename, ".ann") != NULL)  //annotation file
    {
        if (argc > 2)
            strcpy(outputfilename, argv[2]);
        else
        {
            strcpy(outputfilename, inputfilename);
            strcat(outputfilename, "_msp.txt");
        }

        ifstream file(inputfilename, ios::in);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open the file!" << std::endl;
            return 1;
        }

        //generate feature list
        CFeatureList* FeatureList = new CFeatureList();
        float* FeatureIntensity = new float[FeatureList->m_NumFeatures];

        //read .ann file, generating and saving one spectrum a time
        char charline[1024];
        file.getline(charline, 1023, '\n');
        while (charline[0] != '>')
            file.getline(charline, 1023, '\n');

        int b;
        char sequence[64] = "";

        int i, index, NumResidues, charge, CollisionEnergy;
        float intensity, sum;
        char* c;

        using namespace std;
        ofstream fileout(outputfilename, ios::out);

        CMS2Spectrum spectrum(0, NULL, NULL, sequence, 0, 0.0f, 0.0f, FeatureList, FeatureIntensity);
        char buffer[16];
        while (strlen(charline) > 0)
        {
            if (charline[0] == '>')
            {
                charge = 0;
                CollisionEnergy = 0;

                sscanf(charline, ">length=%d;", &NumResidues);
                c = strstr(charline, "sequence=") + 9;
                b = 0;
                while (c[b] != ';')
                {
                    sequence[b] = c[b];
                    b++;
                }
                sequence[b] = '\0';

                c = strstr(c, "charge=") + 7;
                sscanf(c, "%d;", &charge);

                c = strstr(c, "NCE=") + 4;
                sscanf(c, "%d", &CollisionEnergy);

                //get feature intensity
                for (i = 0; i < FeatureList->m_NumFeatures; i++)
                    FeatureIntensity[i] = 0.0f;

                file.getline(charline, 1023, '\n');
                while (charline[0] != '>' && strlen(charline) > 0)
                {
                    if (sscanf(charline, "%d,%f", &index, &intensity) == 2 && intensity > 0.0f)
                        FeatureIntensity[index] = intensity;
                    file.getline(charline, 1023, '\n');
                }

                if (charline[0] == '>' || strlen(charline) == 0) //Complete this spectrum
                {
                    spectrum.SetFeatures(sequence, charge, (float)CollisionEnergy, 0.0f, FeatureList, FeatureIntensity);
                    spectrum.Annotate();    //use the same function to annotate and to generate spectrum
                    
                    fileout << "Name: " << spectrum.m_sequence << '/' << spectrum.m_charge << '\n';
                    fileout << "Comment: Charge=" << spectrum.m_charge << " NCE=" << (int)(spectrum.m_CollisionEnergy + 0.5f) << '\n';
                    fileout << "Num peaks: " << spectrum.m_points << '\n';

                    //normalize
                    sum = 0.0f;
                    for (i = 0; i < spectrum.m_points; i++)
                        sum += spectrum.m_intensity[i];

                    for (i = 0; i < spectrum.m_points; i++)
                    {
                        sprintf(buffer, "%.4f", spectrum.m_mass[i]);
                        fileout << buffer << '\t' << spectrum.m_intensity[i]/sum*1.0e7f << '\n';
                    }
                    fileout << '\n';
                }
            }
        }
        fileout.close();
        delete[] FeatureIntensity;
        delete FeatureList;
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
