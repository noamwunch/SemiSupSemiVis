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
void root_tree_to_txt_with_rave_primvert(const char *inputFile,
                      bool dijet,
                      bool veto_isolep,
                      double PT_min, double PT_max,
                      double Eta_min, double Eta_max,
                      double Mjj_min, double Mjj_max,
                      double ystar_max,
                      double dRjetsMax, const char *result,
                      int bkg_PID=5)
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
    bool p1sig, p2sig, p1bkg, p2bkg;
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

    cout << factory.hasBeamspot() << endl;
    rave::Point3D beam_point = rave::Point3D(0.0, 0.0 ,0.0);
    rave::Covariance3D beam_cov = rave::Covariance3D(5, 0.0, 0.0, 5, 0.0, 5);
    rave::Ellipsoid3D beamspot = rave::Ellipsoid3D(beam_point, beam_cov);
    factory.setBeamSpot(beamspot);
    cout << factory.hasBeamspot() << endl;

    // Loop over all events (except first one)
    Long64_t entry;
    Long64_t n_pass = 0;
    Long64_t n_pass_vetoisolep = 0;
    Long64_t n_pass_dijet = 0;
    Long64_t n_pass_mjj = 0;
    Long64_t n_pass_pt1 = 0;
    Long64_t n_pass_pt2 = 0;
    Long64_t n_pass_eta1 = 0;
    Long64_t n_pass_eta2 = 0;
    Long64_t n_pass_ystar = 0;

    for (entry = 1; entry < allEntries; ++entry) {
        // Load Event
        treeReader->ReadEntry(entry);

        // Loop over particles to find initial dark/QCD partons
        int sig_PID = 4900101;

        p1sig = false;
        p2sig = false;
        p1bkg = false;
        p2bkg = false;
        int i;
        for(i = 0; i < branchParticle->GetEntriesFast(); ++i)
        {
            //Read information particle
            object = (TObject*) branchParticle->At(i);

            //Check information
            particle = (GenParticle*) object;

            if(particle->PID == sig_PID && !p1sig)
            {
                EtaP1 = particle->Eta;
                PhiP1 = particle->Phi;
                PTP1  = particle->PT;
                p1sig = true;
            }

            if(particle->PID == -sig_PID && !p2sig)
            {
                EtaP2 = particle->Eta;
                PhiP2 = particle->Phi;
                PTP2  = particle->PT;
                p2sig = true;
            }

            if(particle->PID == bkg_PID && !p1bkg)
            {
                EtaP1 = particle->Eta;
                PhiP1 = particle->Phi;
                PTP1  = particle->PT;
                p1bkg = true;
            }

            if(particle->PID == -bkg_PID && !p2bkg)
            {
                EtaP2 = particle->Eta;
                PhiP2 = particle->Phi;
                PTP2  = particle->PT;
                p2bkg = true;
            }
        }

        // Get leading jets
        double EtaJ[2] = {-1000, -1000};
        double PhiJ[2] = {0, 0};
        double PTJ[2] = {0, 0};
        double MJ[2] = {0, 0};
        bool JetJ[2] = {false, false};
        j = 0;
        while (j < 2 && j < branchJet->GetEntriesFast()) {
            //Get jet
            jet = (Jet *) branchJet->At(j);
            //Save information
            EtaJ[j] = jet->Eta;
            PhiJ[j] = jet->Phi;
            PTJ[j] = jet->PT;
            MJ[j] = jet->Mass;
            JetJ[j] = true;
            //Increment
            j++;
        }

        // Calculate Mjj
        double Mjj = calc_Mjj(PTJ[0], EtaJ[0], PhiJ[0], PTJ[1], EtaJ[1], PhiJ[1]);

        // Cuts
        bool pass_vetoisolep = true;
        bool pass_dijet = true;
        bool pass_mjj = true;
        bool pass_pt1 = true;
        bool pass_pt2 = true;
        bool pass_eta1 = true;
        bool pass_eta2 = true;
        bool pass_ystar = true;
        if(((branchElectron->GetEntriesFast()>0) || (branchMuon->GetEntriesFast()>0)) && (veto_isolep==true)){ // Isolated lepton veto
            pass_vetoisolep = false;
        }
        else{
            n_pass_vetoisolep += 1;
        }

        if ((JetJ[1]==false) && (dijet==true)) { //Dijet cut
            pass_dijet = false;
        }
        else{
            n_pass_dijet += 1;
        }

        if ((Mjj<Mjj_min) || (Mjj>Mjj_max)) { //Mjj cut
            pass_mjj = false;
        }
        else{
            n_pass_mjj += 1;
        }

        if ((PTJ[0]<PT_min) || (PTJ[0]>PT_max)) { //jet1 PT cut
            pass_pt1 = false;
        }
        else{
            n_pass_pt1 += 1;
        }

        if ((PTJ[1]<PT_min) || (PTJ[1]>PT_max)) { //jet2 PT cut
            pass_pt2 = false;
        }
        else{
            n_pass_pt2 += 1;
        }

        if ((EtaJ[0]<Eta_min) || (EtaJ[0]>Eta_max)) { //jet1 Eta cut
            pass_eta1 = false;
        }
        else{
            n_pass_eta1 += 1;
        }

        if ((EtaJ[1]<Eta_min) || (EtaJ[1]>Eta_max)) { //jet2 Eta cut
            pass_eta2 = false;
        }
        else{
            n_pass_eta2 += 1;
        }

        if (0.5*abs(EtaJ[1]-EtaJ[0])>ystar_max) { //y* cut
            pass_ystar = false;
        }
        else{
           n_pass_ystar += 1;
        }

        if (!(pass_vetoisolep && pass_dijet && pass_mjj && pass_pt1 && pass_pt2 && pass_eta1 && pass_eta2 && pass_ystar)){
            continue;
        }
        else{
            n_pass += 1;
        }

        // Event info
        myfile << "--  Event " << entry << "  --" << endl;
        met = (MissingET *) branchMissingET->At(0);
        myfile << "    MET: " << met->MET << endl; // Event missing energy
        myfile << "    MJJ: " << Mjj << endl;
        myfile << "    y*:  " << 0.5*abs(EtaJ[1]-EtaJ[0]) << endl;

        //If signal, write parton information
        if(p1sig && p2sig) {
            myfile << "    sig_Parton 1    pT: " << PTP1 << " eta: " << EtaP1 << " phi: " << PhiP1 << endl;
            myfile << "    sig_Parton 2    pT: " << PTP2 << " eta: " << EtaP2 << " phi: " << PhiP2 << endl;
        } else if (p1bkg && p2bkg){
            myfile << "    bkg_Parton 1    pT: " << PTP1 << " eta: " << EtaP1 << " phi: " << PhiP1 << endl;
            myfile << "    bkg_Parton 2    pT: " << PTP2 << " eta: " << EtaP2 << " phi: " << PhiP2 << endl;
        }

        //Write information about leading jets
        double deltaR11, deltaR12, deltaR21, deltaR22, deltaR1_nearest_parton, deltaR2_nearest_parton;
        if (JetJ[0]) {
            myfile << "    Jet 1    pT: " << PTJ[0] << " eta: " << EtaJ[0] << " phi: " << PhiJ[0] << " mass: " << MJ[0];
            deltaR11 = pow(pow(EtaP1 - EtaJ[0], 2) + pow(delta_phi_calculator(PhiP1, PhiJ[0]), 2), 0.5);
            deltaR12 = pow(pow(EtaP2 - EtaJ[0], 2) + pow(delta_phi_calculator(PhiP2, PhiJ[0]), 2), 0.5);
            deltaR1_nearest_parton = min(deltaR11, deltaR12);
            myfile << " dR_closest_parton: " << deltaR1_nearest_parton << endl;
        }
        if (JetJ[1]) {
            myfile << "    Jet 2    pT: " << PTJ[1] << " eta: " << EtaJ[1] << " phi: " << PhiJ[1] << " mass: " << MJ[1];
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
        vector <rave::Track> notinj1_tracks;
        vector <rave::Track> notinj2_tracks;
        for (i = 0; i < branchEFlowTrack->GetEntriesFast(); ++i) {
            // Get track
            track = (Track *) branchEFlowTrack->At(i);
            // Compute deltaR
            EtaT = track->Eta;
            PhiT = track->Phi;
            // Check for distance from both jets
            deltaR1 = pow(pow(EtaT - EtaJ[0], 2) + pow(delta_phi_calculator(PhiT, PhiJ[0]), 2), 0.5);
            deltaR2 = pow(pow(EtaT - EtaJ[1], 2) + pow(delta_phi_calculator(PhiT, PhiJ[1]), 2), 0.5);
            // Convert to cartesian and add to all tracks vector
            rave::Vector6D track6d = TrackConvert(track);
            rave::Covariance6D cov6d = CovConvert(track);
            //Write information accordingly
            if (deltaR1 < dRjetsMax) {
                myfile << 1 << " " << track->PT << " " << track->Eta << " " << track->Phi
                << " " << 1 << " " << track->PID << " " << track->D0 << " " << track->DZ << endl;
                j1_tracks.push_back(rave::Track(track6d, cov6d, track->Charge, 0.0, 0.0));
            }
            else{
                notinj1_tracks.push_back(rave::Track(track6d, cov6d, track->Charge, 0.0, 0.0));
            }

            if (deltaR2 < dRjetsMax) {
                myfile << 2 << " " << track->PT << " " << track->Eta << " " << track->Phi
                << " " << 1 << " " << track->PID << " " << track->D0 << " " << track->DZ << endl;
                j2_tracks.push_back(rave::Track(track6d, cov6d, track->Charge, 0.0, 0.0));
            }
            else{
                notinj2_tracks.push_back(rave::Track(track6d, cov6d, track->Charge, 0.0, 0.0));
            }
        }

        //Write vertex information
        double xp, yp, chisq, vert_D0, vert_Dz;

        //vector <rave::Track> tracks;
        vector < std::pair < float, rave::Track > > tracks;

        myfile << "Jet-number D0 Dz Chi-squared Multiplicity type(4=vertex)" << endl;
        vector <rave::Vertex> j1_vertices = factory.create(notinj1_tracks, j1_tracks, true); // Reconstruct vertices

        for (vector<rave::Vertex>::const_iterator r = j1_vertices.begin(); r != j1_vertices.end(); ++r)
        {
            // Extract vertex info
            xp = (*r).position().x() * 10; //Converting to mm (RAVE produces output in cm)
            yp = (*r).position().y() * 10;
            vert_Dz = (*r).position().z() * 10;
            chisq = (*r).chiSquared();
            tracks = (*r).weightedTracks();
            vert_D0 = pow(pow(xp, 2) + pow(yp, 2), 0.5);

            int vert_mult = 0;
            for (vector<std::pair<float,rave::Track>>::const_iterator t = tracks.begin(); t != tracks.end(); ++t)
            {
                float weight = t -> first;
                if (weight < 0.5){
                continue;
                }

                rave::Track track = t -> second;
                double track_px = track.momentum().x();
                double track_py = track.momentum().y();
                vert_mult = vert_mult + 1;
            }
            myfile << 1 << " " << vert_D0 << " " << vert_Dz << " " << chisq << " " << vert_mult << " " << 4 << endl;
        }

        vector <rave::Vertex> j2_vertices = factory.create(notinj2_tracks, j2_tracks, true); // Reconstruct vertices
        for (vector<rave::Vertex>::const_iterator r = j2_vertices.begin(); r != j2_vertices.end(); ++r) {
            // Extract vertex info
            xp = (*r).position().x() * 10; //Converting to mm (RAVE produces output in cm)
            yp = (*r).position().y() * 10;
            vert_Dz = (*r).position().z() * 10;
            chisq = (*r).chiSquared();
            tracks = (*r).weightedTracks();
            vert_D0 = pow(pow(xp, 2) + pow(yp, 2), 0.5);

            int vert_mult = 0;
            for (vector<std::pair<float,rave::Track>>::const_iterator t = tracks.begin(); t != tracks.end(); ++t)
            {
                float weight = t -> first;
                if (weight < 0.5){
                continue;
                }

                rave::Track track = t -> second;
                double track_px = track.momentum().x();
                double track_py = track.momentum().y();
                vert_mult = vert_mult + 1;
            }
            myfile << 2 << " " << vert_D0 << " " << vert_Dz << " " << chisq << " " << vert_mult << " " << 4 << endl;
        }

    }
    myfile << "Done" << endl;
    myfile << "events total " << allEntries << endl;
    myfile << "passed total " << n_pass << endl;
    myfile << "efficiency total " << n_pass/double(allEntries) << endl;
    myfile << "-----dijet-----" << endl;
    myfile << "passed dijet " << n_pass_dijet << endl;
    myfile << "efficiency dijet " << n_pass_dijet/double(allEntries) << endl;
    myfile << "-----mjj-----" << endl;
    myfile << "passed mjj " << n_pass_mjj << endl;
    myfile << "efficiency mjj " << n_pass_mjj/double(allEntries) << endl;
    myfile << "-----pt1-----" << endl;
    myfile << "passed pt1 " << n_pass_pt1 << endl;
    myfile << "efficiency pt1 " << n_pass_pt1/double(allEntries) << endl;
    myfile << "-----pt2-----" << endl;
    myfile << "passed pt2 " << n_pass_pt2 << endl;
    myfile << "efficiency pt2 " << n_pass_pt2/double(allEntries) << endl;
    myfile << "-----eta1-----" << endl;
    myfile << "passed eta1 " << n_pass_eta1 << endl;
    myfile << "efficiency eta1 " << n_pass_eta1/double(allEntries) << endl;
    myfile << "-----eta2-----" << endl;
    myfile << "passed eta2 " << n_pass_eta2 << endl;
    myfile << "efficiency eta2 " << n_pass_eta2/double(allEntries) << endl;
    myfile << "-----ystar-----" << endl;
    myfile << "passed ystar " << n_pass_ystar << endl;
    myfile << "efficiency ystar " << n_pass_ystar/double(allEntries) << endl;
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    std::cout << "Elapsed time = " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " s" << endl;
    delete treeReader;
    delete chain;

}




