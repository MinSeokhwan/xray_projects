#include "event.hh"

thread_local NSEventAction* globNSEventAction = nullptr;

NSEventAction::NSEventAction(NSRunAction*)
: fRayleighCount(0), fPhotoelectricCount(0), fComptonCount(0)
{
    globNSEventAction = this;
}

NSEventAction::~NSEventAction()
{}

void NSEventAction::BeginOfEventAction(const G4Event* evt)
{
    fRayleighCount = 0;
    fPhotoelectricCount = 0;
    fComptonCount = 0;
    
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

void NSEventAction::EndOfEventAction(const G4Event* evt)
{
    G4int evtID = evt->GetEventID();

    G4cout << "Rayleigh: " << fRayleighCount << G4endl;
    G4cout << "Photoelectric: " << fPhotoelectricCount << G4endl;
    G4cout << "Compton: " << fComptonCount << G4endl;

    G4AnalysisManager *man = G4AnalysisManager::Instance();
    man->FillNtupleIColumn(3, 0, evtID);
    man->FillNtupleIColumn(3, 1, fRayleighCount);
    man->FillNtupleIColumn(3, 2, fPhotoelectricCount);
    man->FillNtupleIColumn(3, 3, fComptonCount);
    man->AddNtupleRow(3);
}
