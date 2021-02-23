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

#if defined(_MSC_VER)
# define RaveDllExport __declspec(dllexport)
#else
# define RaveDllExport
#endif

#ifdef __CLING__
R__LOAD_LIBRARY(/gpfs0/kats/projects/rave/lib/libRaveBase.so)
R__LOAD_LIBRARY(/gpfs0/kats/projects/rave/lib/libRaveCore.so)
R__LOAD_LIBRARY(/gpfs0/kats/projects/rave/lib/libRaveVertex.so)

#include "/gpfs0/kats/projects/rave/include/rave/Version.h"
#include "/gpfs0/kats/projects/rave/include/rave/VertexFactory.h"
#include "/gpfs0/kats/projects/rave/include/rave/Vertex.h"
#include "/gpfs0/kats/projects/rave/include/rave/Track.h"
#include "/gpfs0/kats/projects/rave/include/rave/Covariance6D.h"
#include "/gpfs0/kats/projects/rave/include/rave/Vector6D.h"
#include "/gpfs0/kats/projects/rave/include/rave/ConstantMagneticField.h"

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
    return pow(2 * pt1 * pt2 * (cosh(eta1-eta2) - cos(delta_phi_calculator(phi1, phi2))), 0.5);
}

// Convert 5D delphes track to 6D rave track
rave::Vector6D TrackConvert(Track *track) {
    double eps = track->D0 * 0.1;  // also d0   //Delphes is in mm and rave is in cm
    double z0 = track->DZ * 0.1;  // z0
    double pt = track->PT;
    double phi = track->Phi;              // phi_0 (for us)
    double ctth = track->CtgTheta;
    double q = track->Charge;
    double x = eps * sin(phi);
    double y = -eps * cos(phi);
    double z = z0;
    double px = pt / q * cos(phi);
    double py = pt / q * sin(phi);
    double pz = pt / q * ctth;
    rave::Vector6D track6d(x, y, z, px, py, pz);
    return track6d;
}

// Convert 6D rave to 5D delphes track
vector<double> TrackInvConvert(vector<rave::Track>::const_iterator track6) {
    // Read 6D track
    double x = track6->position().x();
    double y = track6->position().y();
    double z = track6->position().z();
    double px = track6->momentum().x();
    double py = track6->momentum().y();
    double pz = track6->momentum().z();
    double charge = double(track6->charge());
    // Compute 5D track
    double pt = sqrt(pow(px, 2) + pow(py, 2)) * abs(charge);
    double theta = atan2(pt, pz * charge);
    double eta = -log(tan(0.5 * theta));
    double phi = atan2(py * charge, px * charge);
    double d0 = sqrt(pow(x, 2) + pow(y, 2)) * abs(charge) * 10;
    double dz = z * 10;
    vector<double> track5 = {pt, eta, phi, d0, dz};
    return track5;
}

// Convert 5D delphes covariance to 6D rave covariance
rave::Covariance6D CovConvert(Track *track) {
    // Read 5D track
    double pt = track->PT;
    double phi = track->Phi;
    double eta = track->Eta;
    double px = pt * cos(phi);
    double py = pt * sin(phi);
    double pz = pt * sinh(eta);
    double ctth = track->CtgTheta;
    double d0 = track->D0 * 0.1; //epsilon
    double q = double(track->Charge);
    // Read 5D errors
    double deld0 = (track->ErrorD0) * 0.1; //in cm
    double delz0 = (track->ErrorDZ) * 0.1; //in cm
    double delpt = track->ErrorPT; //in GeV/c
    double delphi = track->ErrorPhi;
    double delctth = track->ErrorCtgTheta;
    double deltht = delctth / (1 + ctth * ctth);
    // Compute 5D covariance
    double covd0d0 = deld0 * deld0;
    double covz0z0 = delz0 * delz0;
    double covptpt = delpt * delpt;
    double covphiphi = delphi * delphi;
    double covthth = deltht * deltht;
    // Compute 5D covariance
    double dpxpx = py * py * covphiphi + covptpt * cos(phi) * cos(phi) / (q * q);
    double dpxpy = -px * py * covphiphi + covptpt * cos(phi) * sin(phi) / (q * q);
    double dpxpz = covptpt * cos(phi) * ctth / (q * q);// + 0.00000001;
    double dxpx = -d0 * py * covphiphi * cos(phi);
    double dypx = -d0 * py * covphiphi * sin(phi);
    double dpypy = px * px * covphiphi + covptpt * sin(phi) * sin(phi) / (q * q);
    double dpypz = covptpt * sin(phi) * ctth / (q * q);// + 0.0000001;
    double dxpy = d0 * px * covphiphi * cos(phi);
    double dypy = d0 * px * covphiphi * sin(phi);
    double dypz = 0;
    double dxpz = 0;
    double dzpx = 0;
    double dzpy = 0;
    double dzpz = 0;
    double dpzpz = covptpt * ctth * ctth + pt * pt * covthth * pow((1 + ctth * ctth), 2) / (q * q);
    double dxx = d0 * d0 * cos(phi) * cos(phi) * covphiphi + covd0d0 * sin(phi) * sin(phi);
    double dxy = (-covd0d0 + d0 * d0 * covphiphi) * cos(phi) * sin(phi);
    double dxz = 0;
    double dyy = covd0d0 * cos(phi) * cos(phi) + d0 * d0 * sin(phi) * sin(phi) * covphiphi;
    double dyz = 0;
    double dzz = covz0z0;
    rave::Covariance6D cov6d(dxx, dxy, dxz, dyy, dyz, dzz, dxpx, dxpy, dxpz, dypx, dypy, dypz, dzpx, dzpy, dzpz, dpxpx,
                             dpxpy, dpxpz, dpypy, dpypz, dpzpz);
    return cov6d;
}


//Main code
void DB_root_tree_to_txt_with_rave(const char *inputFile,
                      bool dijet,
                      bool veto_isolep,
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
    gSystem->Load("/gpfs0/kats/projects/rave/lib/libRaveBase");
    gSystem->Load("/gpfs0/kats/projects/rave/lib/libRaveCore");
    gSystem->Load("/gpfs0/kats/projects/rave/lib/libRaveVertex");
    gSystem->Load("/gpfs0/kats/projects/rave/lib/libRaveVertexKinematics");

    // Prepare to read root information
    TChain *chain = new TChain("Delphes");
    chain->Add(inputFile);
    ExRootTreeReader *treeReader = new ExRootTreeReader(chain);
    TClonesArray *branchParticle = treeReader->UseBranch("Particle");
    TClonesArray *branchJet = treeReader->UseBranch("Jet");
    TClonesArray *branchMissingET = treeReader->UseBranch("MissingET");
    TClonesArray *branchElectron = treeReader->UseBranch("Electron");
    TClonesArray *branchMuon = treeReader->UseBranch("Muon");

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
    // Create vertex factory
    float Bz = 2.0;   // Magnetic field
    rave::ConstantMagneticField mfield(0., 0., Bz);
    rave::VertexFactory factory(mfield);
    factory.setDefaultMethod("avr");

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
        double Mjj = calc_Mjj(PTJ[0], EtaJ[0], PhiJ[0], PTJ[1], EtaJ[1], PhiJ[1]);

        // Cuts
        if(((branchElectron->GetEntriesFast()>0) || (branchMuon->GetEntriesFast()>0)) && (veto_isolep==true)){ // Isolated lepton veto
            continue;
        }

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

        //If signal, write parton information
        if(p1Ass && p2Ass) {
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

        //Write consituent information (eflow neutral hadrons, eflow photons, eflow tracks)
        if (JetJ[0])
            myfile << "Jet-number PT Eta Phi type(1='track',2='photon',3='neut_had') PID D0 DZ" << endl;

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

        //Loop over eflow tracks
        vector <rave::Track> j1_tracks;
        vector <rave::Track> j2_tracks;
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
                rave::Vector6D track6d = TrackConvert(track);
                rave::Covariance6D cov6d = CovConvert(track);
                j1_tracks.push_back(rave::Track(track6d, cov6d, track->Charge, 0.0, 0.0));
            }
            if (deltaR2 < dRjetsMax) {
                myfile << 2 << " " << track->PT << " " << track->Eta << " " << track->Phi
                << " " << 1 << " " << track->PID << " " << track->D0 << " " << track->DZ << endl;
                rave::Vector6D track6d = TrackConvert(track);
                rave::Covariance6D cov6d = CovConvert(track);
                j2_tracks.push_back(rave::Track(track6d, cov6d, track->Charge, 0.0, 0.0));
            }
        }

        //Write vertex information
        double xp, yp, zp, chisq, vert_D0, vert_mult;
        //vector <rave::Track> tracks;
        vector < std::pair < float, rave::Track > > tracks;
        myfile << "Jet-number D0 Chi-squared Multiplicity type(4=vertex)" << endl;
        vector <rave::Vertex> j1_vertices = factory.create(j1_tracks); // Reconstruct vertices

        // remove
        double vertexed_track_mult = 0;
        cout << "Jet 1 vertexing multiplicities ev " << entry << endl << endl;
        //remove

        for (vector<rave::Vertex>::const_iterator r = j1_vertices.begin(); r != j1_vertices.end(); ++r)
        {
            // Extract vertex info
            xp = (*r).position().x() * 10; //Converting to mm (RAVE produces output in cm)
            yp = (*r).position().y() * 10;
            zp = (*r).position().z() * 10;
            chisq = (*r).chiSquared();

            weighted_tracks = (*r).weightedTracks();
            tracks = (*r).tracks();

            vert_D0 = pow(pow(xp, 2) + pow(yp, 2), 0.5);
            vert_mult = tracks.size();
            myfile << 1 << " " << vert_D0 << " " << chisq << " " << vert_mult << " " << 4 << endl;

            // remove
            vertexed_track_mult = vertexed_track_mult + vert_mult;
            cout << "vertex multiplicity = " << vert_mult << endl;
            cout << "total vertexed track multiplicity = " << vertexed_track_mult << endl;
            cout << endl << "vertex constituents:" << endl;
            for (vector<rave::Track>::const_iterator t = tracks.begin(); t != tracks.end(); ++t)
            {
                float weight = t.first;
                <rave::Track> track = t.second;
                double track_px = track->momentum().x();
                double track_py = track->momentum().y();
                cout << "track px:" << track_px << " track py:" << track_py << endl;
            }
            cout << endl;
            // remove
        }

        // remove
        cout << endl << "final vertexed multiplicity = " << vertexed_track_mult << endl;
        cout << "track multiplicity = " << j1_tracks.size() << endl;
        if (vertexed_track_mult > j1_tracks.size())
        {
            cout << "ERROR!" << endl;
        }

        cout << endl << "tracker constituents:" << endl;
        for (vector<rave::Track>::const_iterator t = j1_tracks.begin(); t != j1_tracks.end(); ++t)
        {
            double track_px = t->momentum().x();
            double track_py = t->momentum().y();
            cout << "track px:" << track_px << " track py:" << track_py << endl;
        }

        cout << " --------------------------------------------------------------- " << endl << endl << endl;
        // remove

        vector <rave::Vertex> j2_vertices = factory.create(j2_tracks); // Reconstruct vertices
        for (vector<rave::Vertex>::const_iterator r = j2_vertices.begin(); r != j2_vertices.end(); ++r) {
            // Extract vertex info
            xp = (*r).position().x() * 10; //Converting to mm (RAVE produces output in cm)
            yp = (*r).position().y() * 10;
            zp = (*r).position().z() * 10;
            chisq = (*r).chiSquared();
            vert_D0 = pow(pow(xp, 2) + pow(yp, 2), 0.5);
            vert_mult = (*r).tracks().size();
            myfile << 2 << " " << vert_D0 << " " << chisq << " " << vert_mult << " " << 4 << endl;
        }

    }
    myfile << "Done" << endl;
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    std::cout << "Elapsed time = " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " s" << endl;
    delete treeReader;
    delete chain;
}




