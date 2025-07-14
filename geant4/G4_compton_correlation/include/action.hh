#ifndef ACTION_HH
#define ACTION_HH

#include "G4VUserActionInitialization.hh"

#include "generator.hh"
#include "run.hh"
#include "event.hh"
#include "stepping.hh"

class NSActionInitialization : public G4VUserActionInitialization
{
public:
    NSActionInitialization();
    ~NSActionInitialization();
    
    virtual void BuildForMaster() const;
    virtual void Build() const;
};

#endif