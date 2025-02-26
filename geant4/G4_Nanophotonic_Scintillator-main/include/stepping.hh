#ifndef STEPPING_HH
#define STEPPING_HH

#include "G4UserSteppingAction.hh"
#include "G4Step.hh"
#include "G4Gamma.hh"
#include "G4Electron.hh"
#include "G4AnalysisManager.hh"
#include "G4RunManager.hh"
#include "G4OpBoundaryProcess.hh"
#include "G4AffineTransform.hh"

#include "construction.hh"
#include "event.hh"

class NSSteppingAction : public G4UserSteppingAction
{
public:
    NSSteppingAction(NSEventAction* eventAction);
    ~NSSteppingAction();
    
    virtual void UserSteppingAction(const G4Step*);
    
private:
    NSEventAction *fEventAction;
};

#endif