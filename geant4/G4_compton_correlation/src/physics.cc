#include "physics.hh"

NSPhysicsList::NSPhysicsList() 
{
    RegisterPhysics (new G4EmStandardPhysics_option4());
    RegisterPhysics (new G4OpticalPhysics());
}

NSPhysicsList::~NSPhysicsList() {}
