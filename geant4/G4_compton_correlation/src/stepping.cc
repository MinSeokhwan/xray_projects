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
    
    G4StepPoint* preStep = step->GetPreStepPoint();
    G4VPhysicalVolume* physVolPre = preStep->GetPhysicalVolume();
    G4String volumeNamePre = physVolPre->GetName();
    
    G4StepPoint* postStep = step->GetPostStepPoint();
    G4VPhysicalVolume* physVolPost = postStep->GetPhysicalVolume();
    G4String volumeNamePost;
    if (physVolPost)
    {
        volumeNamePost = physVolPost->GetName();
    } else
    {
        volumeNamePost = "none";
    }
    
    // G4cout << "PreVol" << volumeNamePre << " | PostVol" << volumeNamePost << G4endl;
    
    if (volumeNamePre == "physWorld" && volumeNamePost == "physDetector")
    {
        G4String creatorProcess = track->GetCreatorProcess() != nullptr? track->GetCreatorProcess()->GetProcessName() : "none";
        G4double vX = track->GetVertexPosition().x();
        G4double vY = track->GetVertexPosition().y();
        G4double vZ = track->GetVertexPosition().z();
    
        G4StepPoint *stepPoint;
        if (particleType == "gamma")
        {
            stepPoint = postStep; // Photoelectric effect and Compton interaction are recorded at the post step
        } else if (particleType == "opticalphoton")
        {
            stepPoint = preStep; // Recording the optical photon creation event
        } else if (particleType == "lepton") // electrons
        {
            stepPoint = postStep;
        }
    
        G4int evt = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
        G4AnalysisManager *man = G4AnalysisManager::Instance();
    
        G4ThreeVector posPhoton = stepPoint->GetPosition();
        G4ThreeVector momPhoton = stepPoint->GetMomentum();
    
        G4double wlen = (1.239841939*eV/momPhoton.mag())*1E+03;
        
        man->FillNtupleIColumn(0, 0, evt);
        man->FillNtupleDColumn(0, 1, wlen);
        man->FillNtupleSColumn(0, 2, particleType);
        //man->FillNtupleSColumn(0, 3, creatorProcess);
        //man->FillNtupleDColumn(0, 4, posPhoton[0]);
        //man->FillNtupleDColumn(0, 5, posPhoton[1]);
        //man->FillNtupleDColumn(0, 6, posPhoton[2]);
        //man->FillNtupleDColumn(0, 7, momPhoton[0]);
        //man->FillNtupleDColumn(0, 8, momPhoton[1]);
        //man->FillNtupleDColumn(0, 9, momPhoton[2]);
        //man->FillNtupleDColumn(0, 10, vX);
        //man->FillNtupleDColumn(0, 11, vY);
        //man->FillNtupleDColumn(0, 12, vZ);
        man->AddNtupleRow(0);
    
        const G4VTouchable *touchable = postStep->GetTouchable();
        G4VPhysicalVolume *physVol = touchable->GetVolume();
        G4ThreeVector posDetector = physVol->GetTranslation();
        
        //G4cout << "x: " << posDetector[0] << " y: " << posDetector[1] << G4endl;
        
        man->FillNtupleIColumn(1, 0, evt);
        //man->FillNtupleDColumn(1, 1, posDetector[0]);
        //man->FillNtupleDColumn(1, 2, posDetector[1]);
        man->FillNtupleDColumn(1, 1, posDetector[2]);
        man->AddNtupleRow(1);
    
        if (particleType != "gamma")
            track->SetTrackStatus(fStopAndKill);
    }
     
    if (particleType == "gamma" && postStep)
    {
        const G4VProcess* proc = postStep->GetProcessDefinedStep();
        
        if (proc && globNSEventAction)
        {
            G4String procName = proc->GetProcessName();
            
            if (procName == "rayl")
                globNSEventAction->IncrementRayleighCount();
            else if (procName == "phot")
            {
                G4cout << "IncrementPhotoelectricCount" << G4endl;
                globNSEventAction->IncrementPhotoelectricCount();
            }
            else if (procName == "compt")
            {
                G4cout << "IncrementComptonCount" << G4endl;
                globNSEventAction->IncrementComptonCount();
            }
        }
    }
    
    if (particleType == "opticalphoton")
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