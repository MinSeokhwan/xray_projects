#include "event.hh"

NSEventAction::NSEventAction(NSRunAction*)
{
    fEdep = 0.;
}

NSEventAction::~NSEventAction()
{}

void NSEventAction::BeginOfEventAction(const G4Event* evt)
{
    fEdep = 0.;
    
    G4int evtID = evt->GetEventID();
    G4ThreeVector posSrc = evt->GetPrimaryVertex()->GetPosition();
    G4PrimaryParticle* srcXray = evt->GetPrimaryVertex()->GetPrimary();
    G4double eneSrc = srcXray->GetKineticEnergy();
    
    G4AnalysisManager *man = G4AnalysisManager::Instance();
    man->FillNtupleIColumn(2, 0, evtID);
    man->FillNtupleDColumn(2, 1, eneSrc);
    man->FillNtupleDColumn(2, 2, posSrc[0]);
    man->FillNtupleDColumn(2, 3, posSrc[1]);
    man->AddNtupleRow(2);
}

void NSEventAction::EndOfEventAction(const G4Event*)
{
    // #ifndef G4MULTITHREADED
    //     G4cout << "Energy deposition: " << fEdep << G4endl;
    // #endif

    //G4AnalysisManager *man = G4AnalysisManager::Instance();
    //man->FillNtupleDColumn(2, 0, fEdep);
    //man->AddNtupleRow(2);
}
