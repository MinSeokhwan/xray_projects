#include "run.hh"

NSRunAction::NSRunAction()
{
    G4AnalysisManager *man = G4AnalysisManager::Instance();

//    man->CreateNtuple("Photons", "Photons"); // photons that hit the detector
//    man->CreateNtupleIColumn("fEvent"); // event ID
//    man->CreateNtupleDColumn("fX"); // x position
//    man->CreateNtupleDColumn("fY"); // y position
//    man->CreateNtupleDColumn("fZ"); // z position
//    man->CreateNtupleDColumn("fT"); // time
//    man->CreateNtupleDColumn("fWlen"); // wavelength
//    man->CreateNtupleSColumn("fType"); // gamma(source photon), opticalphoton(generated photon), lepton(electron)
//    man->CreateNtupleDColumn("fTrackLength"); // total path length (?)
//    man->CreateNtupleSColumn("fMaterial"); // material in which the photon was generated
//    man->CreateNtupleSColumn("fProcess"); // process through which the photon was generated
//    man->CreateNtupleIColumn("fStepStatusNumber");
//    man->CreateNtupleIColumn("fProcessType");
//    man->CreateNtupleIColumn("fSubProcessType");
//    man->CreateNtupleSColumn("fCreatorProcess");
//    man->CreateNtupleIColumn("ftrackID");
//    man->CreateNtupleIColumn("fParentTrackID");
//    man->FinishNtuple(0);
    
    man->CreateNtuple("Photons", "Photons"); // photons that hit the detector
    man->CreateNtupleIColumn("fEvent"); // event ID
    man->CreateNtupleDColumn("fWlen"); // wavelength
    man->CreateNtupleSColumn("particleType");
    man->CreateNtupleSColumn("fCreatorProcess");
    man->CreateNtupleDColumn("fX"); // x position
    man->CreateNtupleDColumn("fY"); // y position
    man->CreateNtupleDColumn("fZ"); // z position
    man->CreateNtupleDColumn("pX"); // x momentum
    man->CreateNtupleDColumn("pY"); // y momentum
    man->CreateNtupleDColumn("pZ"); // z momentum
    man->CreateNtupleDColumn("vX"); // vertex x position
    man->CreateNtupleDColumn("vY"); // vertex y position
    man->CreateNtupleDColumn("vZ"); // vertex z position
    man->FinishNtuple(0);

//    man->CreateNtuple("Hits", "Hits"); // detector perspective
//    man->CreateNtupleIColumn("fEvent"); // event ID
//    man->CreateNtupleDColumn("fX"); // detector x position
//    man->CreateNtupleDColumn("fY"); // detector y position
//    man->CreateNtupleDColumn("fZ"); // detector z position
//    man->FinishNtuple(1);
    
    man->CreateNtuple("Hits", "Hits"); // detector perspective
    man->CreateNtupleIColumn("fEvent"); // event ID
    man->CreateNtupleDColumn("fX"); // detector x position
    man->CreateNtupleDColumn("fY"); // detector y position
    //man->CreateNtupleDColumn("fZ"); // detector z position
    man->FinishNtuple(1);

//    man->CreateNtuple("Scoring", "Scoring");
//    man->CreateNtupleDColumn("fEdep"); // energy deposition
//    man->FinishNtuple(2);

    man->CreateNtuple("Source", "Source"); // xray data
    man->CreateNtupleIColumn("fEvent"); // event ID
    man->CreateNtupleDColumn("fEnergy"); // initial xray energy
    man->CreateNtupleDColumn("fX"); // initial xray x-coordinate
    man->CreateNtupleDColumn("fY"); // initial xray y-coordinate
    man->FinishNtuple(2);
    
    man->CreateNtuple("Process", "Process");
    man->CreateNtupleIColumn("fEvent");
    man->CreateNtupleIColumn("Rayleigh");
    man->CreateNtupleIColumn("Photoelectric");
    man->CreateNtupleIColumn("Compton");
    man->FinishNtuple(3);

    root_file_name = "output.root";
    fMessenger = new G4GenericMessenger(this, "/system/", "System params");
    fMessenger->DeclareProperty("root_file_name", root_file_name, "Name of the output root file");

}

NSRunAction::~NSRunAction()
{}

void NSRunAction::BeginOfRunAction(const G4Run* run)
{
    G4AnalysisManager *man = G4AnalysisManager::Instance();
    //if (root_file_name == "output.root")
    //{
    //    G4int runID = run->GetRunID();
    //    std::stringstream strRunID;
    //    strRunID << runID;
    //    root_file_name = "output"+strRunID.str()+".root";
    //}
    man->OpenFile(root_file_name);
}

void NSRunAction::EndOfRunAction(const G4Run*)
{
    G4AnalysisManager *man = G4AnalysisManager::Instance();
    man->Write();
    man->CloseFile();
}
