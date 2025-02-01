#include "detector.hh"

NSSensitiveDetector::NSSensitiveDetector(G4String name, bool use_detector) : G4VSensitiveDetector(name)
{
    useDetector = use_detector;
    // quEff = new G4PhysicsOrderedFreeVector();
    // std::ifstream datafile;
    // datafile.open("eff.dat");
    // while(1)
    // {
    //     G4double wlen, queff;
    //     datafile >> wlen >> queff;
    //     if(datafile.eof())
    //         break;
    //     G4cout << wlen << " " << queff << std::endl;
    //     quEff->InsertValues(wlen, queff/100.);
    // }
    // datafile.close();
    // quEff->SetSpline(false);
}

NSSensitiveDetector::~NSSensitiveDetector()
{}

G4bool NSSensitiveDetector::ProcessHits(G4Step *aStep, G4TouchableHistory *ROhist)
{
    // Do net register events due to entering the volume
    G4StepPoint *preStepPoint = aStep->GetPreStepPoint(); // when the photon enters the detector volume
    G4StepPoint *postStepPoint = aStep->GetPostStepPoint(); // when the photon leaves the detector volume

    if (!preStepPoint || !postStepPoint) return true; // don't do anything when the photon is just travelling inside the detector (intermediate timestep between entering and leaving)
    
    if (!useDetector)
    {
        const bool particleTraversedVolume = preStepPoint->GetStepStatus() == fGeomBoundary && postStepPoint->GetStepStatus() == fGeomBoundary;
        if (particleTraversedVolume) return true;
    }

    G4StepPoint *stepPoint = preStepPoint;
    if (preStepPoint->GetStepStatus() == fGeomBoundary)
    {
        stepPoint = postStepPoint;
    }

    G4Track *track = aStep->GetTrack();
    G4String creatorProcess = track->GetCreatorProcess() != nullptr? track->GetCreatorProcess()->GetProcessName() : "none";
    G4ParticleDefinition* particleDef = track->GetDefinition();
    G4String particleType = particleDef->GetParticleType();
    const G4int trackID = track->GetTrackID();
    const G4int parentTrackID = track->GetParentID();
    G4double vZ = track->GetVertexPosition().z();
    
    if (particleType == "gamma")
    {
        if (postStepPoint->GetStepStatus() == fGeomBoundary || postStepPoint->GetStepStatus() == fWorldBoundary) return true;
        stepPoint = postStepPoint; // Photoelectric effect and Compton interaction are recorded at the post step
        if (useDetector)
        {
            track->SetTrackStatus(fStopAndKill);
        }
    }
    if (particleType == "opticalphoton")
    {
        stepPoint = preStepPoint; // Recording the optical photon creation event
        if (stepPoint == nullptr) return true;
        if (useDetector)
        {
            track->SetTrackStatus(fStopAndKill);
        } else {
            if (stepPoint->GetStepStatus() == fGeomBoundary || stepPoint->GetStepStatus() == fWorldBoundary) return true;
        }
    }
    if (particleType == "lepton") // electrons
    {
        stepPoint = postStepPoint;
        if (stepPoint->GetStepStatus() == fGeomBoundary || stepPoint->GetStepStatus() == fWorldBoundary) return true;
    }

    auto process = stepPoint->GetProcessDefinedStep();
    G4String processName = process != nullptr ? process->GetProcessName() : "none";
    G4ProcessType processType = process != nullptr ? process->GetProcessType() : fNotDefined;
    G4int subProcessType = process != nullptr ? process->GetProcessSubType() : -1;

    G4int evt = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
    G4AnalysisManager *man = G4AnalysisManager::Instance();

    G4ThreeVector posPhoton = stepPoint->GetPosition();
    G4ThreeVector momPhoton = stepPoint->GetMomentum();
    float depositedEnergy = aStep->GetTotalEnergyDeposit();
    float trackLength = track->GetTrackLength();

    G4double time = stepPoint->GetGlobalTime();
    G4double wlen = (1.239841939*eV/momPhoton.mag())*1E+03;
    const G4VTouchable *touchable = preStepPoint->GetTouchable();
    G4int copyNo = touchable->GetCopyNumber();
    G4String material = touchable->GetVolume()->GetLogicalVolume()->GetMaterial()->GetName();

    // #ifndef G4MULTITHREADED // Printing only in single thread mode
        // G4cout << "Photon position: " << posPhoton << G4endl;
        // G4cout << "Copy number: " << copyNo << G4endl;
        // G4cout << "Detector position: " << posDetector << G4endl;
    	// G4cout << "Photon wavelength: " << wlen << G4endl;
    	// G4cout << "Particle Type: " << particleType << G4endl;
    	// G4cout << "Creator: " << creatorProcess << G4endl;
    // #endif

//    man->FillNtupleIColumn(0, 0, evt);
//    man->FillNtupleDColumn(0, 1, posPhoton[0]/mm);
//    man->FillNtupleDColumn(0, 2, posPhoton[1]/mm);
//    man->FillNtupleDColumn(0, 3, posPhoton[2]/mm);
//    man->FillNtupleDColumn(0, 4, time);
//    man->FillNtupleDColumn(0, 5, wlen);
//    man->FillNtupleSColumn(0, 6, particleType);
//    man->FillNtupleDColumn(0, 7, trackLength);
//    man->FillNtupleSColumn(0, 8, material);
//    man->FillNtupleSColumn(0, 9, processName);
//    man->FillNtupleIColumn(0, 10, stepPoint->GetStepStatus());
//    man->FillNtupleIColumn(0, 11, processType);
//    man->FillNtupleIColumn(0, 12, subProcessType);
//    man->FillNtupleSColumn(0, 13, creatorProcess);
//    man->FillNtupleIColumn(0, 14, trackID);
//    man->FillNtupleIColumn(0, 15, parentTrackID);
//    man->AddNtupleRow(0);
    
    man->FillNtupleIColumn(0, 0, evt);
    man->FillNtupleDColumn(0, 1, wlen);
    man->FillNtupleSColumn(0, 2, creatorProcess);
    man->FillNtupleDColumn(0, 3, momPhoton[0]);
    man->FillNtupleDColumn(0, 4, momPhoton[1]);
    man->FillNtupleDColumn(0, 5, momPhoton[2]);
    man->FillNtupleDColumn(0, 6, vZ);
    man->AddNtupleRow(0);

    G4VPhysicalVolume *physVol = touchable->GetVolume();
    G4ThreeVector posDetector = physVol->GetTranslation();
    // if(G4UniformRand() < quEff->Value(wlen)) // Detector quantum efficienct{}
    
//    man->FillNtupleIColumn(1, 0, evt);
//    man->FillNtupleDColumn(1, 1, posDetector[0]);
//    man->FillNtupleDColumn(1, 2, posDetector[1]);
//    man->FillNtupleDColumn(1, 3, posDetector[2]);
//    man->AddNtupleRow(1);
    
    man->FillNtupleIColumn(1, 0, evt);
    man->FillNtupleDColumn(1, 1, posDetector[0]);
    man->FillNtupleDColumn(1, 2, posDetector[1]);
    //man->FillNtupleDColumn(1, 3, posDetector[2]);
    man->AddNtupleRow(1);
    return true;
}