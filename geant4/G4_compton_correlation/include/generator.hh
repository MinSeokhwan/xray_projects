#ifndef GENERATOR_HH
#define GENERATOR_HH

#include "G4VUserPrimaryGeneratorAction.hh"

#include "G4ParticleGun.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"
#include "G4IonTable.hh"
#include "G4Gamma.hh"
#include "G4ChargedGeantino.hh"
#include "Randomize.hh"
#include "G4GeneralParticleSource.hh"
//#include "G4GenericMessenger.hh"
//#include "G4RunManager.hh"
//#include "G4AnalysisManager.hh"

//#include "run.hh"

class NSPrimaryGenerator : public G4VUserPrimaryGeneratorAction
{
    public:
        NSPrimaryGenerator();
        ~NSPrimaryGenerator();
    
        virtual void GeneratePrimaries(G4Event*);
    
    private:
        //G4ParticleGun *fParticleGun;
        G4GeneralParticleSource *fParticleGun;
        //G4GenericMessenger *fMessenger;
};

#endif