#ifndef CONSTRUCTION_HH
#define CONSTRUCTION_HH

#include "G4VUserDetectorConstruction.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4PVPlacement.hh"
#include "G4NistManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4GenericMessenger.hh"
#include "G4OpticalSurface.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4Colour.hh"
#include "G4VisAttributes.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4UserLimits.hh"
#include "G4PhysicsOrderedFreeVector.hh"

#include "detector.hh"
#include <G4Types.hh>
#include <G4String.hh>

class NSDetectorConstruction : public G4VUserDetectorConstruction
{
public:
    NSDetectorConstruction();
    ~NSDetectorConstruction();

    std::vector<G4LogicalVolume*> GetScoringVolume() const { return fScoringVolumes; }
    std::vector<G4LogicalVolume*> GetSampleVolume() const { return fSampleVolumes; }

    virtual G4VPhysicalVolume *Construct();
    void ConstructSample();
    void ConstructCylindricalPhantom();
    void ConstructMulticolorScintillator();
    void ConstructSensitiveDetector();
    G4double GetTotalThickness();

private:
    G4Navigator* navigator;

    G4Box *solidWorld, *solidDetector, *solidSample, *solidSoftTissue;
    G4Tubs *solidMuscle, *solidBone, *solidIBlood, *solidGdBlood;
    G4LogicalVolume *logicWorld, *logicDetector, *logicSample, *logicAir, *logicSoftTissue, *logicMuscle, *logicBone, *logicIBlood, *logicGdBlood;
    G4VPhysicalVolume *physWorld, *physDetector, *physSample, *physAir, *physSoftTissue, *physMuscle, *physBone, *physIBlood, *physGdBlood;

    G4Box* solidMultilayerNS;
    G4LogicalVolume* logicMultilayerNS;
    std::vector<G4VPhysicalVolume*> physMultilayerNSArray;
    G4OpticalSurface *mirrorSurface;

    G4UserLimits* userLimits;
    float fStepLimit;

    G4Material *worldMat, *detMat, *YAGCe, *ZnSeTe, *BaF2, *LaCl3Ce, *LYSOCe, *CsITl, *GSOCe, *NaITl, *GadoxTb, *SiO2, *TiO2;
    
    G4Material *AMTeeth, *AMMineralBone, *AMHumeriUpper, *AMHumeriLower, *AMLowerArmBone, *AMHandBone, *AMClavicles, *AMCranium, *AMFemoraUpper, *AMFemoraLower, *AMLowerLeg, *AMFoot, *AMMandible, *AMPelvis;
    G4Material *AMRibs, *AMScapulae, *AMCervicalSpine, *AMThoracicSpine, *AMLumbarSpine, *AMSacrum, *AMSternum, *AMHumeriFemoraUpperMedullaryCavity, *AMHumeriFemoraLowerMedullaryCavity;
    G4Material *AMLowerArmMedullaryCavity, *AMLowerLegMedullaryCavity, *AMCartilage, *AMSkin, *AMBlood, *AMMuscle, *AMLiver, *AMPancreas, *AMBrain, *AMHeart, *AMEyes, *AMKidneys, *AMStomach;
    G4Material *AMSmallIntestine, *AMLargeIntestine, *AMSpleen, *AMThyroid, *AMUrinaryBladder, *AMTestes, *AMAdrenals, *AMOesophagus, *AMGallbladder, *AMProstate, *AMLymph, *AMBreast, *AMAdiposeTissue;
    G4Material *AMLung, *AMGastroIntestinalContents, *AMUrine, *AMAir, *AMIBlood, *AMGdBlood;
    
    G4Material *AFTeeth, *AFMineralBone, *AFHumeriUpper, *AFHumeriLower, *AFLowerArmBone, *AFHandBone, *AFClavicles, *AFCranium, *AFFemoraUpper, *AFFemoraLower, *AFLowerLeg, *AFFoot, *AFMandible, *AFPelvis;
    G4Material *AFRibs, *AFScapulae, *AFCervicalSpine, *AFThoracicSpine, *AFLumbarSpine, *AFSacrum, *AFSternum, *AFHumeriFemoraUpperMedullaryCavity, *AFHumeriFemoraLowerMedullaryCavity;
    G4Material *AFLowerArmMedullaryCavity, *AFLowerLegMedullaryCavity, *AFCartilage, *AFSkin, *AFBlood, *AFMuscle, *AFLiver, *AFPancreas, *AFBrain, *AFHeart, *AFEyes, *AFKidneys, *AFStomach;
    G4Material *AFSmallIntestine, *AFLargeIntestine, *AFSpleen, *AFThyroid, *AFUrinaryBladder, *AFOvaries, *AFAdrenals, *AFOesophagus, *AFGallbladder, *AFUterus, *AFLymph, *AFBreast, *AFAdiposeTissue;
    G4Material *AFLung, *AFGastroIntestinalContents, *AFUrine, *AFAir, *AFIodinatedBlood;
    G4Material *sampleMat;
    G4Element *Y, *Al, *O, *Zn, *Se, *Ba, *F, *La, *Cl, *Lu, *Si, *Cs, *I, *H, *B, *C, *N, *Na, *Mg, *P, *S, *K, *Ca, *Fe, *Gd, *Ti;
    
    G4PhysicsOrderedFreeVector *effDet;
    
    G4PhysicsOrderedFreeVector *fraction;
    
    G4PhysicsOrderedFreeVector *rindexWorld, *rindexDet, *rindexYAGCe, *rindexZnSeTe, *rindexBaF2, *rindexLaCl3Ce, *rindexLYSOCe, *rindexCsITl, *rindexGSOCe, *rindexNaITl, *rindexGadoxTb;
    G4PhysicsOrderedFreeVector *rindexSiO2, *rindexTiO2;
    
    G4PhysicsOrderedFreeVector *rindexAMTeeth, *rindexAMMineralBone, *rindexAMHumeriUpper, *rindexAMHumeriLower, *rindexAMLowerArmBone, *rindexAMHandBone, *rindexAMClavicles, *rindexAMCranium;
    G4PhysicsOrderedFreeVector *rindexAMFemoraUpper, *rindexAMFemoraLower, *rindexAMLowerLeg, *rindexAMFoot, *rindexAMMandible, *rindexAMPelvis;
    G4PhysicsOrderedFreeVector *rindexAMRibs, *rindexAMScapulae, *rindexAMCervicalSpine, *rindexAMThoracicSpine, *rindexAMLumbarSpine, *rindexAMSacrum, *rindexAMSternum;
    G4PhysicsOrderedFreeVector *rindexAMHumeriFemoraUpperMedullaryCavity, *rindexAMHumeriFemoraLowerMedullaryCavity;
    G4PhysicsOrderedFreeVector *rindexAMLowerArmMedullaryCavity, *rindexAMLowerLegMedullaryCavity, *rindexAMCartilage, *rindexAMSkin, *rindexAMBlood, *rindexAMMuscle, *rindexAMLiver, *rindexAMPancreas;
    G4PhysicsOrderedFreeVector *rindexAMBrain, *rindexAMHeart, *rindexAMEyes, *rindexAMKidneys, *rindexAMStomach;
    G4PhysicsOrderedFreeVector *rindexAMSmallIntestine, *rindexAMLargeIntestine, *rindexAMSpleen, *rindexAMThyroid, *rindexAMUrinaryBladder, *rindexAMTestes, *rindexAMAdrenals, *rindexAMOesophagus;
    G4PhysicsOrderedFreeVector *rindexAMGallbladder, *rindexAMProstate, *rindexAMLymph, *rindexAMBreast, *rindexAMAdiposeTissue;
    G4PhysicsOrderedFreeVector *rindexAMLung, *rindexAMGastroIntestinalContents, *rindexAMUrine, *rindexAMAir, *rindexAMIBlood, *rindexAMGdBlood;
    
    G4PhysicsOrderedFreeVector *rindexAFTeeth, *rindexAFMineralBone, *rindexAFHumeriUpper, *rindexAFHumeriLower, *rindexAFLowerArmBone, *rindexAFHandBone, *rindexAFClavicles, *rindexAFCranium;
    G4PhysicsOrderedFreeVector *rindexAFFemoraUpper, *rindexAFFemoraLower, *rindexAFLowerLeg, *rindexAFFoot, *rindexAFMandible, *rindexAFPelvis;
    G4PhysicsOrderedFreeVector *rindexAFRibs, *rindexAFScapulae, *rindexAFCervicalSpine, *rindexAFThoracicSpine, *rindexAFLumbarSpine, *rindexAFSacrum, *rindexAFSternum;
    G4PhysicsOrderedFreeVector *rindexAFHumeriFemoraUpperMedullaryCavity, *rindexAFHumeriFemoraLowerMedullaryCavity;
    G4PhysicsOrderedFreeVector *rindexAFLowerArmMedullaryCavity, *rindexAFLowerLegMedullaryCavity, *rindexAFCartilage, *rindexAFSkin, *rindexAFBlood, *rindexAFMuscle, *rindexAFLiver, *rindexAFPancreas;
    G4PhysicsOrderedFreeVector *rindexAFBrain, *rindexAFHeart, *rindexAFEyes, *rindexAFKidneys, *rindexAFStomach;
    G4PhysicsOrderedFreeVector *rindexAFSmallIntestine, *rindexAFLargeIntestine, *rindexAFSpleen, *rindexAFThyroid, *rindexAFUrinaryBladder, *rindexAFOvaries, *rindexAFAdrenals, *rindexAFOesophagus;
    G4PhysicsOrderedFreeVector *rindexAFGallbladder, *rindexAFUterus, *rindexAFLymph, *rindexAFBreast, *rindexAFAdiposeTissue;
    G4PhysicsOrderedFreeVector *rindexAFLung, *rindexAFGastroIntestinalContents, *rindexAFUrine, *rindexAFAir, *rindexAFIodinatedBlood;
    
    G4PhysicsOrderedFreeVector *absLengthDet, *absLengthYAGCe, *absLengthZnSeTe, *absLengthBaF2, *absLengthLaCl3Ce, *absLengthLYSOCe, *absLengthCsITl, *absLengthGSOCe, *absLengthNaITl, *absLengthGadoxTb;
    G4PhysicsOrderedFreeVector *absLengthSiO2, *absLengthTiO2;
    
    G4PhysicsOrderedFreeVector *absLengthAMTeeth, *absLengthAMMineralBone, *absLengthAMHumeriUpper, *absLengthAMHumeriLower, *absLengthAMLowerArmBone, *absLengthAMHandBone, *absLengthAMClavicles;
    G4PhysicsOrderedFreeVector *absLengthAMCranium, *absLengthAMFemoraUpper, *absLengthAMFemoraLower, *absLengthAMLowerLeg, *absLengthAMFoot, *absLengthAMMandible, *absLengthAMPelvis;
    G4PhysicsOrderedFreeVector *absLengthAMRibs, *absLengthAMScapulae, *absLengthAMCervicalSpine, *absLengthAMThoracicSpine, *absLengthAMLumbarSpine, *absLengthAMSacrum, *absLengthAMSternum;
    G4PhysicsOrderedFreeVector *absLengthAMHumeriFemoraUpperMedullaryCavity, *absLengthAMHumeriFemoraLowerMedullaryCavity;
    G4PhysicsOrderedFreeVector *absLengthAMLowerArmMedullaryCavity, *absLengthAMLowerLegMedullaryCavity, *absLengthAMCartilage, *absLengthAMSkin, *absLengthAMBlood, *absLengthAMMuscle, *absLengthAMLiver;
    G4PhysicsOrderedFreeVector *absLengthAMPancreas, *absLengthAMBrain, *absLengthAMHeart, *absLengthAMEyes, *absLengthAMKidneys, *absLengthAMStomach;
    G4PhysicsOrderedFreeVector *absLengthAMSmallIntestine, *absLengthAMLargeIntestine, *absLengthAMSpleen, *absLengthAMThyroid, *absLengthAMUrinaryBladder, *absLengthAMTestes, *absLengthAMAdrenals;
    G4PhysicsOrderedFreeVector *absLengthAMOesophagus, *absLengthAMGallbladder, *absLengthAMProstate, *absLengthAMLymph, *absLengthAMBreast, *absLengthAMAdiposeTissue;
    G4PhysicsOrderedFreeVector *absLengthAMLung, *absLengthAMGastroIntestinalContents, *absLengthAMUrine, *absLengthAMAir, *absLengthAMIBlood, *absLengthAMGdBlood;
    
    G4PhysicsOrderedFreeVector *absLengthAFTeeth, *absLengthAFMineralBone, *absLengthAFHumeriUpper, *absLengthAFHumeriLower, *absLengthAFLowerArmBone, *absLengthAFHandBone, *absLengthAFClavicles;
    G4PhysicsOrderedFreeVector *absLengthAFCranium, *absLengthAFFemoraUpper, *absLengthAFFemoraLower, *absLengthAFLowerLeg, *absLengthAFFoot, *absLengthAFMandible, *absLengthAFPelvis;
    G4PhysicsOrderedFreeVector *absLengthAFRibs, *absLengthAFScapulae, *absLengthAFCervicalSpine, *absLengthAFThoracicSpine, *absLengthAFLumbarSpine, *absLengthAFSacrum, *absLengthAFSternum;
    G4PhysicsOrderedFreeVector *absLengthAFHumeriFemoraUpperMedullaryCavity, *absLengthAFHumeriFemoraLowerMedullaryCavity;
    G4PhysicsOrderedFreeVector *absLengthAFLowerArmMedullaryCavity, *absLengthAFLowerLegMedullaryCavity, *absLengthAFCartilage, *absLengthAFSkin, *absLengthAFBlood, *absLengthAFMuscle, *absLengthAFLiver;
    G4PhysicsOrderedFreeVector *absLengthAFPancreas, *absLengthAFBrain, *absLengthAFHeart, *absLengthAFEyes, *absLengthAFKidneys, *absLengthAFStomach;
    G4PhysicsOrderedFreeVector *absLengthAFSmallIntestine, *absLengthAFLargeIntestine, *absLengthAFSpleen, *absLengthAFThyroid, *absLengthAFUrinaryBladder, *absLengthAFOvaries, *absLengthAFAdrenals;
    G4PhysicsOrderedFreeVector *absLengthAFOesophagus, *absLengthAFGallbladder, *absLengthAFUterus, *absLengthAFLymph, *absLengthAFBreast, *absLengthAFAdiposeTissue;
    G4PhysicsOrderedFreeVector *absLengthAFLung, *absLengthAFGastroIntestinalContents, *absLengthAFUrine, *absLengthAFAir, *absLengthAFIodinatedBlood;

    G4OpticalSurface *opticalSurfaceWorld, *opticalSurfaceDet, *opticalSurfaceYAGCe, *opticalSurfaceZnSeTe, *opticalSurfaceBaF2, *opticalSurfaceLaCl3Ce, *opticalSurfaceLYSOCe, *opticalSurfaceCsITl;
    G4OpticalSurface *opticalSurfaceNaITl, *opticalSurfaceGadoxTb, *opticalSurfaceSiO2, *opticalSurfaceTiO2, *opticalSurfaceGSOCe;

    void DefineMaterials();
    void DefineElements(G4NistManager *nist);
    void DefineOpticalSurface(G4MaterialPropertiesTable* mpt, G4OpticalSurface* opticalSurface, G4String opticalSurfaceName);

    void DefineWorld(G4NistManager *nist);
    void DefineDetector(G4NistManager *nist);
    void DefineYAGCe();
    void DefineZnSeTe();
    void DefineBaF2();
    void DefineLaCl3Ce();
    void DefineLYSOCe();
    void DefineCsITl();
    void DefineGSOCe();
    void DefineNaITl();
    void DefineGadoxTb();
    void DefineSiO2();
    void DefineTiO2();
    void DefineAMBioMedia();
    void DefineAFBioMedia();
    
    virtual void ConstructSDandField();

    G4GenericMessenger *fMessenger;
    std::vector<G4LogicalVolume*> fScoringVolumes;
    std::vector<G4LogicalVolume*> fSampleVolumes;
    G4double xWorld, yWorld, zWorld, gapSampleScint, gapScintDet;
    
    G4bool constructDetectors, constructTopDetector;
    G4int nDetX, nDetY;
    G4double xDet, yDet, detectorDepth, detectorX, detectorY;

    G4int nLayers;
    G4double xScint, yScint;
    G4double scintillatorThickness1, scintillatorThickness2, scintillatorThickness3;
    G4int scintillatorMaterial1, scintillatorMaterial2, scintillatorMaterial3;
    G4bool angleFilter;
    G4int angleFilterStart;
    
    G4int sampleID, organIDij;
    G4String sampleName;
    G4double xSample, ySample, zSample, voxelX, voxelY, voxelZ;
    G4int nSampleXall, nSampleX, nSampleY, nSampleZ, indSampleX, indSampleY, indSampleZ;
    G4double IConcentration, GdConcentration;
	
    std::vector<G4double> layerThicknesses;
    std::vector<G4Material*> layerMaterials;
    std::vector<G4int> layerMaterialName;

    bool checkDetectorsOverlaps;
};

#endif
