// MS2FeatureConverter.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "CMS2Spectrum.h"

extern int bOffset[];
extern int yOffset[];

/*
This program converts a .msp file to a .ann file or vise versa

When the inputfilename is a ".msp" file, it does the following
1. Reads the .msp file (containing peptide MS/MS spectra) and save the selected spectra into a .ms2 file
2. Annotate each selected spectrum and save the extracted feature vector into a .ann file
3. Save the Dictionary file
4. Export some annotated mass offsets as examples

When the inputfilename is a ".ann" file, the program converts each feature vector into a spectrum and save the spectra into a .msp file
*/

int main(int argc, char* argv[])
{
    argc = 2;
    if (argc < 2)
    {
        std::cout << "Please specify file name\n";
        return 1;
    }

    using namespace std;

//  char* inputfilename = argv[1];

    //convert .msp file (spectra) to .ann file (annotation) or vise versa
    char inputfilename[512] = "C:\\Users\\zhang\\Documents\\MSData\\UniSpec_Datasets3\\Predicted\\FullMS2FormerPredict_TestUniq202277_202312.ann.txt";

    char outputfilename[512];
    char* c, *c1;
    //if .msp file (MS/MS spectra), perform annotation/feature extraction, then export the .ann file (outputfilename) and .ms2 file (outputfilename2 - selected spectra with only necessary information)
    if (strstr(inputfilename, ".msp") != NULL)  
    {
        if (argc > 2)
            strcpy(outputfilename, argv[2]);
        else
        {
            //replace .msp with .ann
            strcpy(outputfilename, inputfilename);
            c = strstr(outputfilename, ".msp");
            c[1] = 'a';
            c[2] = 'n';
            c[3] = 'n';
        }

        char outputfilename2[512]; //for selected spectrum
        strcpy(outputfilename2, inputfilename); //rename to .ms2 file
        c = strstr(outputfilename2, ".msp");
        c[3] = '2';

        ifstream file(inputfilename, ios::in);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open the file!" << std::endl;
            return 1;
        }

        //generate feature list (Dictionary)
        CFeatureList *FeatureList = new CFeatureList(); //generate feature list (Dictionary) here
        float* FeatureIntensity = new float[FeatureList->m_NumFeatures];
        char Dictionaryfilename[512];
        strcpy(Dictionaryfilename, inputfilename);
        char *temp = strrchr(Dictionaryfilename, '\\');
        strcpy(temp, "\\Dictionary.txt");
        FeatureList->SaveAs(Dictionaryfilename);

        //generate output file
        ofstream file_out(outputfilename, ios::out);
        file_out << "Dictionary size = " << FeatureList->m_NumFeatures << '\n';
        file_out.close();

        ofstream file_out2(outputfilename2, ios::out);  //open a new file so that the old file will be overwritten
        file_out2.close();

        char charline[1024];
        file.getline(charline, 1023, '\n');

        int b;
        int length = (int)strlen(charline);
        char sequence[128] = "";

        int mod_count, index;
        char residue, mod[256];
        int keep = 1, charge = 0;
        long i, NumIons;
        float PrecursorMz=0.0f, CollisionEnergy=0.0f;
        float StartMass, EndMass, StartMass1, EndMass1, CollisionEnergy1;
        float* mass, * intensity;
        double Mass0 = 0.0;   //peptide mass


        int NLcount[OFFSETSIZE], fraglen, neutral;
        char fragname, header[256];
        float bMaxNL[OFFSETSIZE], yMaxNL[OFFSETSIZE], bTotalNL[OFFSETSIZE], yTotalNL[OFFSETSIZE];
        string bFragIonNL[OFFSETSIZE], bHeaderNL[OFFSETSIZE], yFragIonNL[OFFSETSIZE], yHeaderNL[OFFSETSIZE];
        memset(NLcount, 0, OFFSETSIZE * sizeof(int));
        for (i = 0; i < OFFSETSIZE; i++)
        {
            bMaxNL[i] = 0.0f;
            yMaxNL[i] = 0.0f;
            bTotalNL[i] = 0.0f;
            yTotalNL[i] = 0.0f;
        }

        //total intensity of each feature
        float* TotalFeatureIntensity = new float[FeatureList->m_NumFeatures];
        for (i = 0; i < FeatureList->m_NumFeatures; i++)
            TotalFeatureIntensity[i] = 0.0f;

        //total intensity of different charges
        float TotalChargeIntensity[7];
        for (i = 0; i <= MAXCHARGE; i++)
            TotalChargeIntensity[i] = 0.0f;

        int SpecCount = 0, z;
        while (!(length == 0 && strlen(sequence) == 0))// && SpecCount < 10)
        {
            if (strstr(charline, "Name:") == charline)
            {
                //set value if known
                CollisionEnergy = 0.0f;
                StartMass = 0.0f;
                EndMass = 0.0f;

                charge = 0;
                Mass0 = 0.0;
                PrecursorMz = 0.0f;

                b = 6;
                while (charline[b] != '/')
                {
                    sequence[b - 6] = charline[b];
                    b++;
                }
                sequence[b-6] = '\0';
                sscanf(charline + b, "/%d", &charge);

                //read modification from parenthesis (only takes JUOsty)
                mod_count = 0;
                c = strchr(charline, '(');
                while (c != NULL && keep)
                {
                    sscanf(c, "(%d,%c,%s", &index, &residue, mod);
                    if((c1 = strchr(mod, ')')) != NULL)
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
            }
            else if (strstr(charline, "InstrumentModel:") == charline)
            {
                if (strstr(charline + 17, "Velos") != NULL || strstr(charline + 17, "Elite") != NULL)
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
                
                if ((c = strstr(charline, "@hcd")) != NULL && strlen(sequence) > 0)
                {
                    if (sscanf(c + 4, "%f [%f-%f]", &CollisionEnergy1, &StartMass1, &EndMass1) == 3 && StartMass1 > 10.0f && StartMass1 < 500.0f && EndMass1 > StartMass1)
                    {
                        CollisionEnergy = CollisionEnergy1;
                        StartMass = StartMass1;
                        EndMass = EndMass1;
                    }
                }
                if ((c = strstr(charline, "NCE=")) != NULL && strlen(sequence) > 0)
                {
                    CollisionEnergy = (float)atof(c + 4);
                }
                if ((c = strstr(charline, "Mods=")) != NULL && c[6] == '/' && strlen(sequence) > 0)
                {
                    if (sscanf(c + 5, "%d", &mod_count) == 1 && mod_count > 0)
                    {
                        //read modification from / (only takes JUOsty)
                        mod_count = 0;
                        c = strchr(c+6, '/');
                        while (c != NULL && keep)
                        {
                            sscanf(c, "/%d,%c,%s", &index, &residue, mod);
                            if ((c1 = strchr(mod, '/')) != NULL)
                                *c1 = '\0';
                            if (sequence[index] == residue)
                            {
                                if (sequence[index] == 'C' && strcmp(mod, "Carbamidomethyl") == 0)
                                    sequence[index] = 'U';
                                else if (sequence[index] == 'C' && strcmp(mod, "Carboxymethyl") == 0)
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

                            c = strchr(c + 1, '/');
                        }
                    }
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

                    if (charge != 0 && CollisionEnergy > 0.0f && keep)
                    {
                        //send spectrum to CMS2Spectrum class
                        if (strlen(sequence) <= FeatureList->m_MaxNumResidues)
                        {
                            for (i = 0; i < FeatureList->m_NumFeatures; i++)
                                FeatureIntensity[i] = 0.0f;
                            CMS2Spectrum spectrum(NumIons, mass, intensity, sequence, charge, CollisionEnergy, PrecursorMz, FeatureList, FeatureIntensity, StartMass, EndMass);
                            spectrum.ExportSpectrum(outputfilename2);
                            spectrum.Annotate();
                            spectrum.ExportAnnotations(outputfilename);
                            SpecCount++;

                            //find the most intense example for each neutral loss
                            for (i = 0; i < FeatureList->m_NumFeatures; i++)
                            {                             
                                if (FeatureIntensity[i] > 0.5f)
                                {
                                    //Sum charge states
                                    z = atoi(FeatureList->m_Features[i].c_str()+FeatureList->m_Features[i].length() - 3);
                                    TotalChargeIntensity[z] += FeatureIntensity[i];

                                    TotalFeatureIntensity[i] += FeatureIntensity[i];
                                    if (sscanf(FeatureList->m_Features[i].c_str(), "%c%d-%d", &fragname, &fraglen, &neutral) == 3 && neutral < OFFSETSIZE)
                                    {
                                        NLcount[neutral]++;
                                        if (fragname == 'b')
                                        {
                                            bTotalNL[neutral] += FeatureIntensity[i];
                                            if (bMaxNL[neutral] < FeatureIntensity[i])
                                            {
                                                bMaxNL[neutral] = FeatureIntensity[i];
                                                bFragIonNL[neutral] = FeatureList->m_Features[i];
                                                sprintf(header, "%s (%d+ %.0f%%)", sequence, charge, CollisionEnergy);
                                                bHeaderNL[neutral] = header;
                                            }
                                        }
                                        else if (fragname == 'y')
                                        {
                                            yTotalNL[neutral] += FeatureIntensity[i];
                                            if (yMaxNL[neutral] < FeatureIntensity[i])
                                            {
                                                yMaxNL[neutral] = FeatureIntensity[i];
                                                yFragIonNL[neutral] = FeatureList->m_Features[i];
                                                sprintf(header, "%s (%d+ %.0f%%)", sequence, charge, CollisionEnergy);
                                                yHeaderNL[neutral] = header;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else if (length == 0)   //blank line, finish entry
            {
                //initialize next entry
                sequence[0] = '\0';
                keep = 1;
                CollisionEnergy = 0.0f;
                charge = 0;
                PrecursorMz = 0.0f;
                Mass0 = 0.0;
                NumIons = 0;
            }
            file.getline(charline, 1023, '\n');
            length = (int)strlen(charline);
        }

        //the following code exports some additional annotation information for information purpose only
        //find an example of each mass offset and export them into the Examples.txt file
        char Examplefilename[512];
        strcpy(Examplefilename, inputfilename);
        temp = strrchr(Examplefilename, '\\');
        strcpy(temp, "\\Examples.txt");

        ofstream file_out3(Examplefilename, ios::out);
        for (i=0; i<NUM_B_OFFSETS; i++)
            file_out3 << "b-" << bOffset[i] << '\t' << bTotalNL[bOffset[i]] << '\t' << bFragIonNL[bOffset[i]] << '\t' << bMaxNL[bOffset[i]] << '\t' << bHeaderNL[bOffset[i]] << '\n';
        for (i = 0; i < NUM_Y_OFFSETS; i++)
            file_out3 << "y-" << yOffset[i] << '\t' << yTotalNL[yOffset[i]] << '\t' << yFragIonNL[yOffset[i]] << '\t' << yMaxNL[yOffset[i]] << '\t' << yHeaderNL[yOffset[i]] << '\n';
        file_out3.close();

        //save total feature intensity
        char FeatureIntensityfilename[512];
        strcpy(FeatureIntensityfilename, inputfilename);
        temp = strrchr(FeatureIntensityfilename, '\\');
        strcpy(temp, "\\FeatureTotalIntensities.txt");

        ofstream file_out4(FeatureIntensityfilename, ios::out);
        for (i = 0; i < FeatureList->m_NumFeatures; i++)
            file_out4 << FeatureList->m_Features[i] << '\t' << TotalFeatureIntensity[i] << '\n';

        file_out4 << '\n';
        file_out4 << "Charge Distribution\n";
        for (i = 1; i <= MAXCHARGE; i++)
            file_out4 << i << '\t' << TotalChargeIntensity[i] << '\n';
        file_out4.close();

        delete[] TotalFeatureIntensity;
        delete [] FeatureIntensity;
        delete FeatureList;
        file.close();

        std::cout << SpecCount << " spectra converted.\n";
    }
    else if (strstr(inputfilename, ".ann") != NULL)  //if annotation file, convert it back to spectrum
    {
        if (argc > 2)
            strcpy(outputfilename, argv[2]);
        else
        {
            strcpy(outputfilename, inputfilename);
            strcat(outputfilename, ".msp.txt");
        }

        ifstream file(inputfilename, ios::in);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open the file!" << std::endl;
            return 1;
        }

        //generate feature list
        CFeatureList* FeatureList = new CFeatureList(); //this is called during both annotation and spectrum generation, therefore they are the same list
        float* FeatureIntensity = new float[FeatureList->m_NumFeatures];

        //read .ann file, generating and saving one spectrum a time
        char charline[1024];
        file.getline(charline, 1023, '\n');
        while (charline[0] != '>')
            file.getline(charline, 1023, '\n');

        int b;
        char sequence[64] = "";

        int i, index, NumResidues, charge;
        float CollisionEnergy, StartMass, EndMass;
        float intensity, sum;
        char* c;

        using namespace std;
        ofstream fileout(outputfilename, ios::out);

        CMS2Spectrum spectrum(0, NULL, NULL, sequence, 0, 0.0f, 0.0f, FeatureList, FeatureIntensity, 0.0f, 0.0f);   //The same object use for all. Spectrum specific info applied in SetFeatures()
        char buffer[16];
        float FeatureSum = 0.0f, Factor;
        while (strlen(charline) > 0)
        {
            if (charline[0] == '>')
            {
                charge = 0;
                CollisionEnergy = 0.0f;
                StartMass = 0.0f;
                EndMass = 0.0f;

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
                sscanf(c, "%f", &CollisionEnergy);

                c = strstr(c, "mass range=") + 11;
                sscanf(c, "[%f-%f]", &StartMass, &EndMass);

                //get feature intensity
                for (i = 0; i < FeatureList->m_NumFeatures; i++)
                    FeatureIntensity[i] = 0.0f;
                FeatureSum = 0.0f;

                file.getline(charline, 1023, '\n');
                while (charline[0] != '>' && strlen(charline) > 0)
                {
                    if (sscanf(charline, "%d,%f", &index, &intensity) == 2 && intensity > 0.0f)
                    {
                        FeatureIntensity[index] = intensity;
                        FeatureSum += intensity;
                    }
                    file.getline(charline, 1023, '\n');
                }

                //normalize FeatureIntensity to 1e7
                Factor = 1.0e7f / FeatureSum;
                for (i = 0; i < FeatureList->m_NumFeatures; i++)
                    FeatureIntensity[i] *= Factor;

                if (charline[0] == '>' || strlen(charline) == 0) //Complete this spectrum
                {
                    spectrum.SetFeatures(sequence, charge, (float)CollisionEnergy, 0.0f, FeatureList, FeatureIntensity, StartMass, EndMass);
                    spectrum.Annotate();    //use the same function to annotate and to generate spectrum
                    spectrum.GenerateSpectrum();

                    fileout << "Name: " << spectrum.m_sequence << '/' << spectrum.m_charge << '\n';
                    fileout << "Comment: Charge=" << spectrum.m_charge << " NCE=" << (int)(spectrum.m_CollisionEnergy + 0.5f) << '\n';
                    fileout << "Num peaks: " << spectrum.m_points << '\n';

                    for (i = 0; i < spectrum.m_points; i++)
                    {
                        sprintf(buffer, "%.4f", spectrum.m_mass[i]);
                        fileout << buffer << '\t' << spectrum.m_intensity[i] << '\n';
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

