#ifndef DETECTOR_HH
#define DETECTOR_HH

#include "G4VSensitiveDetector.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4AnalysisManager.hh"
#include "G4GenericMessenger.hh"

class NSSensitiveDetector : public G4VSensitiveDetector
{
public:
    NSSensitiveDetector(G4String, bool use_detector);
    ~NSSensitiveDetector();
    
private:
    bool useDetector;

    virtual G4bool ProcessHits(G4Step *, G4TouchableHistory *);
    
    // G4PhysicsOrderedFreeVector *quEff; // Detector quantum efficienct
};

#endif