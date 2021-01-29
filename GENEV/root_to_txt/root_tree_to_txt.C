/*************************************************************/
/*                                                           */
/*                Information classifier                     */
/*                                                           */
/*************************************************************/

/* Explanation
This macro extracts the relevant information for machine learning and puts in a .txt file.

Comments:
- Based on Example3.c
*/


#ifdef __CLING__

R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"

#include <iostream>
#include <fstream>
#include <chrono>
#define PI 3.14159265359

#else

class ExRootTreeReader;

#endif

using namespace std;

// Calculate difference between two phi angles
double delta_phi_calculator(double phi1, double phi2) {
    return (abs(phi1 - phi2) <= PI) ? abs(phi1 - phi2) : (2 * PI - abs(phi1 - phi2));
}

double calc_Mjj(double pt1, double eta1, double phi1, double pt2, double eta2, double phi2) {
    return 2 * pt1 * pt2 * (cosh(eta1-eta2) - cos(delta_phi_calculator(phi1, phi2)));
}


//Main code
void root_tree_to_txt(const char *inputFile,
                      bool dijet,
                      double PT_min, double PT_max,
                      double Eta_min, double Eta_max,
                      double Mjj_min, double Mjj_max,
                      double dRjetsMax, const char *result)
{
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    //Prepare to write
    ofstream myfile;
    myfile.open(result);

    //Load Delphes libraries
    gSystem->Load("libDelphes");

    // Prepare to read root information
    TChain *chain = new TChain("Delphes");
    chain->Add(inputFile);
    ExRootTreeReader *treeReader = new ExRootTreeReader(chain);
    TClonesArray *branchParticle = treeReader->UseBranch("Particle");
    TClonesArray *branchJet = treeReader->UseBranch("Jet");
    TClonesArray *branchTrack = treeReader->UseBranch("Track");
    //TClonesArray *branchTower    = treeReader->UseBranch("Tower");
    TClonesArray *branchMissingET = treeReader->UseBranch("MissingET");

    TClonesArray *branchEFlowTrack = treeReader->UseBranch("EFlowTrack");
    TClonesArray *branchEFlowPhoton = treeReader->UseBranch("EFlowPhoton");
    TClonesArray *branchEFlowNeutralHadron = treeReader->UseBranch("EFlowNeutralHadron");

    //File info
    Long64_t allEntries = treeReader->GetEntries();
    myfile << "** Chain contains " << allEntries << " events" << endl;

    //Define variables
    GenParticle *particle;
    Track *track;
    Tower *tower;
    Jet *jet;
    TObject *object;
    MissingET *met;
    // Define parton variables
    double EtaP1, EtaP2;
    double PhiP1, PhiP2;
    double PTP1, PTP2;
    bool p1Ass, p2Ass;
    // Define closest jets temp variables
    double deltaEta1, deltaEta2;
    double deltaPhi1, deltaPhi2;
    double deltaR1t, deltaR2t;
    double deltaR1;
    double deltaR2;
    // Define closest jets variables
    int j;
    // Define track variables
    double EtaT;
    double PhiT;

    // Loop over all events (except first one)
    Long64_t entry;
    for (entry = 1; entry < allEntries; ++entry) {
        // Load Event
        treeReader->ReadEntry(entry);

        // Loop over particles to find initial dark partons
        p1Ass = false;
        p2Ass = false;
        int i;
        for(i = 0; i < branchParticle->GetEntriesFast(); ++i)
        {
            //Read information particle
            object = (TObject*) branchParticle->At(i);

            //Check information
            particle = (GenParticle*) object;

            if(particle->PID == 4900101 && !p1Ass)
            {
                EtaP1 = particle->Eta;
                PhiP1 = particle->Phi;
                PTP1  = particle->PT;
                p1Ass = true;
            }

            if(particle->PID == -4900101 && !p2Ass)
            {
                EtaP2 = particle->Eta;
                PhiP2 = particle->Phi;
                PTP2  = particle->PT;
                p2Ass = true;
            }
        }

        // Get leading jets
        double EtaJ[2] = {-1000, -1000};
        double PhiJ[2] = {0, 0};
        double PTJ[2] = {0, 0};
        bool JetJ[2] = {false, false};
        j = 0;
        while (j < 2 && j < branchJet->GetEntriesFast()) {
            //Get jet
            jet = (Jet *) branchJet->At(j);
            //Save information
            EtaJ[j] = jet->Eta;
            PhiJ[j] = jet->Phi;
            PTJ[j] = jet->PT;
            JetJ[j] = true;
            //Increment
            j++;
        }

        // Calculate Mjj
        double Mjj = calc_Mjj(PTJ[0], EtaJ[0], PhiJ[0], PTJ[1], EtaJ[1], PhiJ[1])

        if ((JetJ[1]==false) && (dijet==true)) { //Dijet cut
            continue;
        }

        if ((Mjj<Mjj_min) || (Mjj>Mjj_max)) { //Mjj cut
            continue;
        }

        if ((PTJ[0]<PT_min) || (PTJ[0]>PT_max)) { //jet1 PT cut
            continue;
        }

        if ((PTJ[1]<PT_min) || (PTJ[1]>PT_max)) { //jet2 PT cut
            continue;
        }

        if ((EtaJ[0]<Eta_min) || (EtaJ[0]>Eta_max)) { //jet1 Eta cut
            continue;
        }

        if ((EtaJ[1]<Eta_min) || (EtaJ[1]>Eta_max)) { //jet2 Eta cut
            continue;
        }

        // Event info
        myfile << "--  Event " << entry << "  --" << endl;
        met = (MissingET *) branchMissingET->At(0);
        myfile << "    MET: " << met->MET << endl; // Event missing energy
        myfile << "    MJJ: " << Mjj << endl;

        if(p1Ass && p2Ass) {
            //Write information of the partons
            myfile << "    Parton 1    pT: " << PTP1 << " eta: " << EtaP1 << " phi: " << PhiP1 << endl;
            myfile << "    Parton 2    pT: " << PTP2 << " eta: " << EtaP2 << " phi: " << PhiP2 << endl;
        }

        //Write information about leading jets
        double deltaR11, deltaR12, deltaR21, deltaR22, deltaR1_nearest_parton, deltaR2_nearest_parton;
        if (JetJ[0]) {
            myfile << "    Jet 1    pT: " << PTJ[0] << " eta: " << EtaJ[0] << " phi: " << PhiJ[0];
            deltaR11 = pow(pow(EtaP1 - EtaJ[0], 2) + pow(delta_phi_calculator(PhiP1, PhiJ[0]), 2), 0.5);
            deltaR12 = pow(pow(EtaP2 - EtaJ[0], 2) + pow(delta_phi_calculator(PhiP2, PhiJ[0]), 2), 0.5);
            deltaR1_nearest_parton = min(deltaR11, deltaR12);
            myfile << " dR_closest_parton: " << deltaR1_nearest_parton << endl;
        }
        if (JetJ[1]) {
            myfile << "    Jet 2    pT: " << PTJ[1] << " eta: " << EtaJ[1] << " phi: " << PhiJ[1];
            deltaR21 = pow(pow(EtaP1 - EtaJ[1], 2) + pow(delta_phi_calculator(PhiP1, PhiJ[1]), 2), 0.5);
            deltaR22 = pow(pow(EtaP2 - EtaJ[1], 2) + pow(delta_phi_calculator(PhiP2, PhiJ[1]), 2), 0.5);
            deltaR2_nearest_parton = min(deltaR21, deltaR22);
            myfile << " dR_closest_parton: " << deltaR2_nearest_parton << endl;
        }
        if (JetJ[0])
            myfile << "Jet-number PT Eta Phi type(1='track',2='photon',3='neut_had') PID D0 DZ" << endl;

        //Loop over eflow tracks
        for (i = 0; i < branchEFlowTrack->GetEntriesFast(); ++i) {
            //Get track
            track = (Track *) branchEFlowTrack->At(i);
            // Compute deltaR
            EtaT = track->Eta;
            PhiT = track->Phi;
            //Check for distance from both jets
            deltaR1 = pow(pow(EtaT - EtaJ[0], 2) + pow(delta_phi_calculator(PhiT, PhiJ[0]), 2), 0.5);
            deltaR2 = pow(pow(EtaT - EtaJ[1], 2) + pow(delta_phi_calculator(PhiT, PhiJ[1]), 2), 0.5);
            //Write information accordingly
            if (deltaR1 < dRjetsMax) {
                myfile << 1 << " " << track->PT << " " << track->Eta << " " << track->Phi
                << " " << 1 << " " << track->PID << " " << track->D0 << " " << track->DZ << endl;
            }
            if (deltaR2 < dRjetsMax) {
                myfile << 2 << " " << track->PT << " " << track->Eta << " " << track->Phi
                << " " << 1 << " " << track->PID << " " << track->D0 << " " << track->DZ << endl;
            }
        }

        //Loop over eflow photons
        for (i = 0; i < branchEFlowPhoton->GetEntriesFast(); ++i) {
            //Get tower
            tower = (Tower *) branchEFlowPhoton->At(i);
            // Compute deltaR
            EtaT = tower->Eta;
            PhiT = tower->Phi;
            //Check for distance from both jets
            deltaR1 = pow(pow(EtaT - EtaJ[0], 2) + pow(delta_phi_calculator(PhiT, PhiJ[0]), 2), 0.5);
            deltaR2 = pow(pow(EtaT - EtaJ[1], 2) + pow(delta_phi_calculator(PhiT, PhiJ[1]), 2), 0.5);
            //Write information accordingly
            if (deltaR1 < dRjetsMax) {
                myfile << 1 << " " << tower->ET << " " << tower->Eta << " " << tower->Phi
                << " " << 2 << " " << 0 << " " << 0 << " " << 0 << endl;
            }
            if (deltaR2 < dRjetsMax) {
                myfile << 2 << " " << tower->ET << " " << tower->Eta << " " << tower->Phi
                << " " << 2 << " " << 0 << " " << 0 << " " << 0 << endl;
            }
        }

        //Loop over eflow neutral hadrons
        for (i = 0; i < branchEFlowNeutralHadron->GetEntriesFast(); ++i) {
            //Get tower
            tower = (Tower *) branchEFlowNeutralHadron->At(i);
            // Compute deltaR
            EtaT = tower->Eta;
            PhiT = tower->Phi;
            //Check for distance from both jets
            deltaR1 = pow(pow(EtaT - EtaJ[0], 2) + pow(delta_phi_calculator(PhiT, PhiJ[0]), 2), 0.5);
            deltaR2 = pow(pow(EtaT - EtaJ[1], 2) + pow(delta_phi_calculator(PhiT, PhiJ[1]), 2), 0.5);
            //Write information accordingly
            if (deltaR1 < dRjetsMax) {
                myfile << 1 << " " << tower->ET << " " << tower->Eta << " " << tower->Phi
                << " " << 3 << " " << 1 << " " << 0 << " " << 0 << endl;
            }
            if (deltaR2 < dRjetsMax) {
                myfile << 2 << " " << tower->ET << " " << tower->Eta << " " << tower->Phi
                << " " << 3 << " " << 1 << " " << 0 << " " << 0 << endl;
            }
        }
    }
    myfile << "Done" << endl;
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    std::cout << "Elapsed time = " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " s" << endl;
    delete treeReader;
    delete chain;
}




