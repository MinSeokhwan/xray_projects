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
    
    G4int GetRayleighCount() const { return fRayleighCount; }
    G4int GetComptonCount() const { return fComptonCount; }
    G4int GetPhotoelectricCount() const { return fPhotoelectricCount; }
    
    void IncrementRayleighCount() { ++fRayleighCount; }
    void IncrementPhotoelectricCount() { ++fPhotoelectricCount; }
    void IncrementComptonCount() { ++fComptonCount; }
    
private:
    G4int fComptonCount;
    G4int fRayleighCount;
    G4int fPhotoelectricCount;
};

extern thread_local NSEventAction* globNSEventAction;

#endif