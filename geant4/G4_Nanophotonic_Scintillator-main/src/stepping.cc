#include "stepping.hh"

NSSteppingAction::NSSteppingAction(NSEventAction *eventAction)
{
    fEventAction = eventAction;
}

NSSteppingAction::~NSSteppingAction()
{}

void NSSteppingAction::UserSteppingAction(const G4Step *step)
{   
    G4Track *track = step->GetTrack();
    G4ParticleDefinition* particleDef = track->GetDefinition();
    G4String particleType = particleDef->GetParticleType();
    
    if (particleType == "opticalphoton")
    {
        G4StepPoint* preStep = step->GetPreStepPoint();
        G4StepPoint* postStep = step->GetPostStepPoint();
        
        G4int evt = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
        G4ThreeVector momPhoton = preStep->GetMomentum();
        G4double wlen = (1.239841939*eV/momPhoton.mag())*1E+03;
    
        if (track->GetTrackStatus() == fStopAndKill)
        {
            G4ThreeVector finalPosition = postStep->GetPosition();
            G4VPhysicalVolume* physVol = postStep->GetPhysicalVolume();
            G4String volumeName;
            if (physVol)
            {
                volumeName = physVol->GetName();
            } else {
                volumeName = "none";
            }
            
            G4AnalysisManager *man = G4AnalysisManager::Instance();
            man->FillNtupleIColumn(3, 0, evt);
            man->FillNtupleDColumn(3, 1, finalPosition[2]);
            man->FillNtupleDColumn(3, 2, wlen);
            man->FillNtupleSColumn(3, 3, volumeName);
            man->AddNtupleRow(3);
        }

        if (postStep->GetStepStatus() == fGeomBoundary)
        {
            G4ThreeVector momentumDir = track->GetMomentumDirection();
      
            G4String prePhysVol = preStep->GetPhysicalVolume()->GetName();
            G4String postPhysVol = postStep->GetPhysicalVolume()->GetName();
            
            G4AnalysisManager *man = G4AnalysisManager::Instance();
            man->FillNtupleIColumn(4, 0, evt);
            man->FillNtupleDColumn(4, 1, momentumDir[2]);
            man->FillNtupleDColumn(4, 2, wlen);
            man->FillNtupleSColumn(4, 3, prePhysVol);
            man->FillNtupleSColumn(4, 4, postPhysVol);
            man->AddNtupleRow(4);
        }
    }
    
    if (particleType != "gamma")
    {
        G4LogicalVolume *volume = step->GetPreStepPoint()->GetTouchableHandle()->GetVolume()->GetLogicalVolume();
        const NSDetectorConstruction *detectorConstruction = static_cast<const NSDetectorConstruction*> (G4RunManager::GetRunManager()->GetUserDetectorConstruction());
        std::vector<G4LogicalVolume*> fSampleVolumes = detectorConstruction->GetSampleVolume();
        
        for (auto& fSampleVolume: fSampleVolumes)
        {
            if(volume == fSampleVolume)
            {
                track->SetTrackStatus(fStopAndKill);
                return;
            }
            //G4double edep = step->GetTotalEnergyDeposit();
            //fEventAction->AddEdep(edep);
        }
    }
}