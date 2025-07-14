#ifndef PHYSICS_HH
#define PHYSICS_HH

#include "PhysListEmPenelope.hh"

#include "G4VModularPhysicsList.hh"
#include "G4EmStandardPhysics.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4EmLivermorePhysics.hh"
#include "G4OpticalPhysics.hh"
#include "G4DecayPhysics.hh"
#include "G4ProcessManager.hh"
#include "G4PhysicsListHelper.hh"
#include "G4EmPenelopePhysics.hh"

// gamma
#include "G4PhotoElectricEffect.hh"
#include "G4PenelopePhotoElectricModel.hh"
#include "G4ComptonScattering.hh"
#include "G4PenelopeComptonModel.hh"
#include "G4GammaConversion.hh"
#include "G4PenelopeGammaConversionModel.hh"
#include "G4RayleighScattering.hh" 
#include "G4PenelopeRayleighModel.hh"

#include "G4VPhysicsConstructor.hh"
#include "G4PolarizedCompton.hh"

#include "G4BosonConstructor.hh"
#include "G4ShortLivedConstructor.hh"
#include "G4IonConstructor.hh"
#include "G4LeptonConstructor.hh"
#include "G4ChargedGeantino.hh"
#include "G4Geantino.hh"
#include "G4OpticalPhoton.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"

// deexcitation
#include "G4LossTableManager.hh"
#include "G4UAtomicDeexcitation.hh"

#include "G4StepLimiterPhysics.hh"

class NSPhysicsList : public G4VModularPhysicsList
{
public:
    NSPhysicsList();
    ~NSPhysicsList();
};

#endif