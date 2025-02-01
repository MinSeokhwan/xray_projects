#ifndef EVENT_HH
#define EVENT_HH

#include "G4UserEventAction.hh"
#include "G4Event.hh"
#include "G4AnalysisManager.hh"

#include "run.hh"

class NSEventAction : public G4UserEventAction
{
public:
    NSEventAction(NSRunAction*);
    ~NSEventAction();
    
    virtual void BeginOfEventAction(const G4Event*);
    virtual void EndOfEventAction(const G4Event*);
    
    void AddEdep(G4double edep) { fEdep += edep; } 
    
private:
    G4double fEdep;
};

#endif