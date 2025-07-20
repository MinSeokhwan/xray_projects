#include "construction.hh"

#include <iostream>
#include <iterator>
#include <fstream>
#include <vector>

NSDetectorConstruction::NSDetectorConstruction()
{
    // World dimensions
    rWorld = 50.; //cm
    zWorld = 10.; //cm
    gapSampleScint = 0.001; //cm
    gapScintDet = 0.001; //cm

    // The overlaps between the detectors should be checked each time the geometry is changed
    // This is set to false by default to avoid performing this verification at each run
    checkDetectorsOverlaps = false;

    // Construct an array of sensitive detectors
    xDet = 100000.; //um
    yDet = 100000.; //um
    nDetX = 1;
    nDetY = 1;
    detectorDepth = 0.1; //um

    // Specify structure type
    rScintCorr = 0.5; //cm
    zScintCorr = 0.1; //cm
    materialScintCorr = 1; // 1: YAGCe  2: ZnSeTe  3: LYSOCe  4: CsITl  5: GSOCe  6: NaITl  7: GadoxTb
    xScintImg = 0.5; //cm
    yScintImg = 0.5; //cm
    zScintImg = 0.1; //cm
    materialScintImg = 3; // 1: YAGCe  2: ZnSeTe  3: LYSOCe  4: CsITl  5: GSOCe  6: NaITl  7: GadoxTb
    
    // Specify Imaging Sample
    sampleID = 0; // 0: None  1: AM  2: AF  3: Cylindrical
    xSample = 0.; //cm
    ySample = 0.; //cm
    zSample = 0.; //cm
    nSampleXall = 254;
    nSampleX = 1;
    nSampleY = 1;
    nSampleZ = 1;
    indSampleX = 0;
    indSampleY = 0;
    indSampleZ = 0;
    IConcentration = 0.01;
    GdConcentration = 0.01;

    // Messenger
    fMessenger = new G4GenericMessenger(this, "/structure/", "Structure construction");
    fMessenger->DeclareProperty("rWorld", rWorld, "Simulation radius");
    fMessenger->DeclareProperty("zWorld", zWorld, "Simulation height (thickness)");
    fMessenger->DeclareProperty("gapSampleScint", gapSampleScint, "Gap between the scintillator and the sample");
    fMessenger->DeclareProperty("gapScintDet", gapScintDet, "Gap between the scintillator and the detector");
    
    fMessenger->DeclareProperty("detectorDepth", detectorDepth, "Depth of the detectors");
    fMessenger->DeclareProperty("xDet", xDet, "Detector x length");
    fMessenger->DeclareProperty("yDet", yDet, "Detector y length");
    fMessenger->DeclareProperty("nDetX", nDetX, "Detector x grid size");
    fMessenger->DeclareProperty("nDetY", nDetY, "Detector y grid size");
    fMessenger->DeclareProperty("checkDetectorsOverlaps", checkDetectorsOverlaps, "Check the detectors for overlap");
    
    fMessenger->DeclareProperty("rScintCorr", rScintCorr, "Correlating scintillator radius");
    fMessenger->DeclareProperty("zScintCorr", zScintCorr, "Correlating scintillator thickness");
    fMessenger->DeclareProperty("materialScintCorr", materialScintCorr, "The material of the correlating scintillator");
    fMessenger->DeclareProperty("xScintImg", xScintImg, "Detecting scintillator x-width");
    fMessenger->DeclareProperty("yScintImg", yScintImg, "Detecting scintillator y-width");
    fMessenger->DeclareProperty("zScintImg", zScintImg, "Detecting scintillator thickness");
    fMessenger->DeclareProperty("materialScintImg", materialScintImg, "The material of the detecting scintillator");
    
    fMessenger->DeclareProperty("sampleID", sampleID, "The phantom to be imaged");
    fMessenger->DeclareProperty("xSample", xSample, "Sample x length");
    fMessenger->DeclareProperty("ySample", ySample, "Sample y length");
    fMessenger->DeclareProperty("zSample", zSample, "Sample z length");
    fMessenger->DeclareProperty("nSampleXall", nSampleXall, "Total number of phantom voxels along x");
    fMessenger->DeclareProperty("nSampleX", nSampleX, "Number of sample voxels along x");
    fMessenger->DeclareProperty("nSampleY", nSampleY, "Number of sample voxels along y");
    fMessenger->DeclareProperty("nSampleZ", nSampleZ, "Number of sample voxels along z");
    fMessenger->DeclareProperty("indSampleX", indSampleX, "Starting voxel index along x");
    fMessenger->DeclareProperty("indSampleY", indSampleY, "Starting voxel index along y");
    fMessenger->DeclareProperty("indSampleZ", indSampleZ, "Starting voxel index along z");
    fMessenger->DeclareProperty("IConcentration", IConcentration, "Relative concentration of Iodine in blood");
    fMessenger->DeclareProperty("GdConcentration", GdConcentration, "Relative concentration of Gadolinium in blood");
}

NSDetectorConstruction::~NSDetectorConstruction()
{
    delete navigator;
}

float wavelenthToeV(float wavelength)
{
    return (1239.84193/(wavelength/nm)) * eV;
}

void DefineScintillator(G4MaterialPropertiesTable* mpt, G4Material* material, const G4int numComponents, G4PhysicsOrderedFreeVector* rindex, G4PhysicsOrderedFreeVector* fraction, G4PhysicsOrderedFreeVector* absLength, const float yield, const float resolutionScale, const float timeConstant)
{
    mpt->AddProperty("RINDEX", rindex);
    mpt->AddProperty("ABSLENGTH", absLength);
    mpt->AddConstProperty("SCINTILLATIONYIELD", yield); // num photons per keV
    mpt->AddConstProperty("RESOLUTIONSCALE",resolutionScale); // proportional to standard deviation of photon yield (normal dist.)
    mpt->AddProperty("SCINTILLATIONCOMPONENT1", fraction); // normalized emission spectrum
    mpt->AddConstProperty("SCINTILLATIONYIELD1", 1.);
    mpt->AddConstProperty("SCINTILLATIONTIMECONSTANT1", timeConstant); // decay time (ns)
    material->SetMaterialPropertiesTable(mpt);
    material->GetIonisation()->SetBirksConstant(0.126 * mm / MeV);
}

void ReadDataFile(const char* filename, G4PhysicsOrderedFreeVector* vec)
{
    std::ifstream dataFile;
    dataFile.open(filename);
    while(1)
    {
      	G4double col1, col2;
      	dataFile >> col1 >> col2;
        if(std::strstr(filename, "rindex") != nullptr)
        {
      	    vec->InsertValues(wavelenthToeV(col1*nm), col2);
        } else if(std::strstr(filename, "emission") != nullptr)
        {
            vec->InsertValues(wavelenthToeV(col1*nm), col2);
        } else if(std::strstr(filename, "linAttCoeff") != nullptr)
        {
            vec->InsertValues(col1*keV, 1./col2*cm);
        }
        if(dataFile.eof())
      	    break;
    }
    dataFile.close();
}

G4MaterialPropertiesTable* DefineNonScintillatingMaterial(G4Material* material, const G4int numComponents, G4PhysicsOrderedFreeVector* rindex, G4PhysicsOrderedFreeVector* absLength)
{
    G4MaterialPropertiesTable *mpt = new G4MaterialPropertiesTable();
    mpt->AddProperty("RINDEX", rindex);
    mpt->AddProperty("ABSLENGTH", absLength);
    material->SetMaterialPropertiesTable(mpt);
    return mpt;
}

void NSDetectorConstruction::DefineOpticalSurface(G4OpticalSurface* opticalSurface)
{
    opticalSurface->SetType(dielectric_dielectric);
    opticalSurface->SetFinish(polished);
}

void NSDetectorConstruction::DefineElements(G4NistManager *nist)
{
    Y = nist->FindOrBuildElement("Y"); // YAG (Y3Al5O12)
    Al = nist->FindOrBuildElement("Al");
    O = nist->FindOrBuildElement("O");
    
    Zn = nist->FindOrBuildElement("Zn"); // ZnSe
    Se = nist->FindOrBuildElement("Se");
    
    Ba = nist->FindOrBuildElement("Ba"); // BaF2
    F = nist->FindOrBuildElement("F");
    
    La = nist->FindOrBuildElement("La"); // LaCl3
    Cl = nist->FindOrBuildElement("Cl");
    
    Lu = nist->FindOrBuildElement("Lu"); // LYSO (Lu1.8Y0.2SiO5)
    Si = nist->FindOrBuildElement("Si");
    
    Cs = nist->FindOrBuildElement("Cs"); // CsI
    I = nist->FindOrBuildElement("I");
    
    Gd = nist->FindOrBuildElement("Gd"); // GSO (Gd2SiO5)
    
    H = nist->FindOrBuildElement("H");
    B = nist->FindOrBuildElement("B");
    C = nist->FindOrBuildElement("C");
    N = nist->FindOrBuildElement("N");
    Na = nist->FindOrBuildElement("Na");
    Mg = nist->FindOrBuildElement("Mg");
    P = nist->FindOrBuildElement("P");
    S = nist->FindOrBuildElement("S");
    K = nist->FindOrBuildElement("K");
    Ca = nist->FindOrBuildElement("Ca");
    Fe = nist->FindOrBuildElement("Fe");
    Ti = nist->FindOrBuildElement("Ti");
}

void NSDetectorConstruction::DefineWorld(G4NistManager *nist)
{
    worldMat = nist->FindOrBuildMaterial("G4_AIR");
    
    rindexWorld = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexWorld);
    
    G4MaterialPropertiesTable *mptWorld = new G4MaterialPropertiesTable();
    mptWorld->AddProperty("RINDEX", rindexWorld);
    worldMat->SetMaterialPropertiesTable(mptWorld);

    // // Creating the optical surface properties
    opticalSurfaceWorld = new G4OpticalSurface("interfaceSurfaceWorld");
    DefineOpticalSurface(opticalSurfaceWorld);
}

void NSDetectorConstruction::DefineDetector(G4NistManager *nist)
{
    detMat = new G4Material("detMat", 2.328*g/cm3, 1);
    detMat->AddElement(Si, 1);

    rindexDet = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexDet);
    
    absLengthDet = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffDet.dat", absLengthDet);
    
    G4MaterialPropertiesTable *mptDet = DefineNonScintillatingMaterial(detMat, 2, rindexDet, absLengthDet);

    // // Creating the optical surface properties
    opticalSurfaceDet = new G4OpticalSurface("interfaceSurfaceDet");
    DefineOpticalSurface(opticalSurfaceDet);  
}

void NSDetectorConstruction::DefineYAGCe()
{
    YAGCe = new G4Material("YAGCe", 4.55*g/cm3, 3);
    YAGCe->AddElement(Y, 3);
    YAGCe->AddElement(Al, 5);
    YAGCe->AddElement(O, 12);

    fraction = new G4PhysicsOrderedFreeVector();
    ReadDataFile("emissionYAGCe.dat", fraction);
    
    rindexYAGCe = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexYAG.dat", rindexYAGCe);
    
    absLengthYAGCe = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffYAGCe.dat", absLengthYAGCe);
    
    G4MaterialPropertiesTable* mptYAGCe = new G4MaterialPropertiesTable();
    DefineScintillator(mptYAGCe, YAGCe, 3, rindexYAGCe, fraction, absLengthYAGCe, 35./keV, 1., 70.*ns); //35

    // // Creating the optical surface properties
    opticalSurfaceYAGCe = new G4OpticalSurface("interfaceSurfaceYAGCe");
    DefineOpticalSurface(opticalSurfaceYAGCe);
}

void NSDetectorConstruction::DefineZnSeTe()
{
    ZnSeTe = new G4Material("ZnSeTe", 5.42*g/cm3, 2);
    ZnSeTe->AddElement(Zn, 1);
    ZnSeTe->AddElement(Se, 1);
    
    fraction = new G4PhysicsOrderedFreeVector();
    ReadDataFile("emissionZnSeTe.dat", fraction);
    
    rindexZnSeTe = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexZnSe.dat", rindexZnSeTe);
    
    absLengthZnSeTe = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffZnSeTe.dat", absLengthZnSeTe);
    
    G4MaterialPropertiesTable* mptZnSeTe = new G4MaterialPropertiesTable();
    DefineScintillator(mptZnSeTe, ZnSeTe, 2, rindexZnSeTe, fraction, absLengthZnSeTe, 55./keV, 1., 50000.*ns);

    // Creating the optical surface properties
    opticalSurfaceZnSeTe = new G4OpticalSurface("interfaceSurfaceZnSeTe");
    DefineOpticalSurface(opticalSurfaceZnSeTe);
}

void NSDetectorConstruction::DefineBaF2()
{
    BaF2 = new G4Material("BaF2", 4.89*g/cm3, 2);
    BaF2->AddElement(Ba, 1);
    BaF2->AddElement(F, 2);
    
    fraction = new G4PhysicsOrderedFreeVector();
    ReadDataFile("emissionBaF2.dat", fraction);
    
    rindexBaF2 = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexBaF2.dat", rindexBaF2);
    
    absLengthBaF2 = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffBaF2.dat", absLengthBaF2);
    
    G4MaterialPropertiesTable* mptBaF2 = new G4MaterialPropertiesTable();
    DefineScintillator(mptBaF2, BaF2, 2, rindexBaF2, fraction, absLengthBaF2, 10./keV, 1., 630.*ns);

    // Creating the optical surface properties
    opticalSurfaceBaF2 = new G4OpticalSurface("interfaceSurfaceBaF2");
    DefineOpticalSurface(opticalSurfaceBaF2);
}

void NSDetectorConstruction::DefineLaCl3Ce()
{
    LaCl3Ce = new G4Material("LaCl3Ce", 3.86*g/cm3, 2);
    LaCl3Ce->AddElement(La, 1);
    LaCl3Ce->AddElement(Cl, 3);
    
    fraction = new G4PhysicsOrderedFreeVector();
    ReadDataFile("emissionLaCl3Ce.dat", fraction);
    
    rindexLaCl3Ce = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexLaCl3.dat", rindexLaCl3Ce);
    
    absLengthLaCl3Ce = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffLaCl3Ce.dat", absLengthLaCl3Ce);
    
    G4MaterialPropertiesTable* mptLaCl3Ce = new G4MaterialPropertiesTable();
    DefineScintillator(mptLaCl3Ce, LaCl3Ce, 2, rindexLaCl3Ce, fraction, absLengthLaCl3Ce, 49./keV, 1., 28.*ns);

    // Creating the optical surface properties
    opticalSurfaceLaCl3Ce = new G4OpticalSurface("interfaceSurfaceLaCl3Ce");
    DefineOpticalSurface(opticalSurfaceLaCl3Ce);
}

void NSDetectorConstruction::DefineLYSOCe()
{
    LYSOCe = new G4Material("LYSOCe", 7.15*g/cm3, 4);
    LYSOCe->AddElement(Lu, 18);
    LYSOCe->AddElement(Y, 2);
    LYSOCe->AddElement(Si, 10);
    LYSOCe->AddElement(O, 50);
    
    fraction = new G4PhysicsOrderedFreeVector();
    ReadDataFile("emissionLYSOCe.dat", fraction);
    
    rindexLYSOCe = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexLYSO.dat", rindexLYSOCe);
    
    absLengthLYSOCe = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffLYSOCe.dat", absLengthLYSOCe);
    
    G4MaterialPropertiesTable* mptLYSOCe = new G4MaterialPropertiesTable();
    DefineScintillator(mptLYSOCe, LYSOCe, 4, rindexLYSOCe, fraction, absLengthLYSOCe, 25./keV, 1., 40.*ns);

    // Creating the optical surface properties
    opticalSurfaceLYSOCe = new G4OpticalSurface("interfaceSurfaceLYSOCe");
    DefineOpticalSurface(opticalSurfaceLYSOCe);
}

void NSDetectorConstruction::DefineCsITl()
{
    CsITl = new G4Material("CsITl", 4.51*g/cm3, 2);
    CsITl->AddElement(Cs, 1);
    CsITl->AddElement(I, 1);
    
    fraction = new G4PhysicsOrderedFreeVector();
    ReadDataFile("emissionCsITl.dat", fraction);
    
    rindexCsITl = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexCsI.dat", rindexCsITl);
    
    absLengthCsITl = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffCsITl.dat", absLengthCsITl);
    
    G4MaterialPropertiesTable* mptCsITl = new G4MaterialPropertiesTable();
    DefineScintillator(mptCsITl, CsITl, 2, rindexCsITl, fraction, absLengthCsITl, 54./keV, 1., 1000.*ns);

    // Creating the optical surface properties
    opticalSurfaceCsITl = new G4OpticalSurface("interfaceSurfaceCsITl");
    DefineOpticalSurface(opticalSurfaceCsITl);
}

void NSDetectorConstruction::DefineGSOCe()
{
    GSOCe = new G4Material("GSOCe", 6.7*g/cm3, 3);
    GSOCe->AddElement(Gd, 2);
    GSOCe->AddElement(Si, 1);
    GSOCe->AddElement(O, 5);
    
    fraction = new G4PhysicsOrderedFreeVector();
    ReadDataFile("emissionGSOCe.dat", fraction);
    
    rindexGSOCe = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexGSO.dat", rindexGSOCe);
    
    absLengthGSOCe = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffGSOCe.dat", absLengthGSOCe);
    
    G4MaterialPropertiesTable* mptGSOCe = new G4MaterialPropertiesTable();
    DefineScintillator(mptGSOCe, GSOCe, 3, rindexGSOCe, fraction, absLengthGSOCe, 10./keV, 1., 50.*ns);

    // Creating the optical surface properties
    opticalSurfaceGSOCe = new G4OpticalSurface("interfaceSurfaceGSOCe");
    DefineOpticalSurface(opticalSurfaceGSOCe);
}

void NSDetectorConstruction::DefineNaITl()
{
    NaITl = new G4Material("NaITl", 3.67*g/cm3, 2);
    NaITl->AddElement(Na, 1);
    NaITl->AddElement(I, 1);
    
    fraction = new G4PhysicsOrderedFreeVector();
    ReadDataFile("emissionNaITl.dat", fraction);
    
    rindexNaITl = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexNaI.dat", rindexNaITl);
    
    absLengthNaITl = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffNaITl.dat", absLengthNaITl);
    
    G4MaterialPropertiesTable* mptNaITl = new G4MaterialPropertiesTable();
    DefineScintillator(mptNaITl, NaITl, 2, rindexNaITl, fraction, absLengthNaITl, 38./keV, 1., 230.*ns);

    // Creating the optical surface properties
    opticalSurfaceNaITl = new G4OpticalSurface("interfaceSurfaceNaITl");
    DefineOpticalSurface(opticalSurfaceNaITl);
}

void NSDetectorConstruction::DefineGadoxTb()
{
    GadoxTb = new G4Material("GadoxTb", 7.34*g/cm3, 3);
    GadoxTb->AddElement(Gd, 2);
    GadoxTb->AddElement(O, 2);
    GadoxTb->AddElement(S, 1);
    
    fraction = new G4PhysicsOrderedFreeVector();
    ReadDataFile("emissionGadoxTb.dat", fraction);
    
    rindexGadoxTb = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexGadox.dat", rindexGadoxTb);
    
    absLengthGadoxTb = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffGadoxTb.dat", absLengthGadoxTb);
    
    G4MaterialPropertiesTable* mptGadoxTb = new G4MaterialPropertiesTable();
    DefineScintillator(mptGadoxTb, GadoxTb, 3, rindexGadoxTb, fraction, absLengthGadoxTb, 60./keV, 1., 1000000.*ns);

    // Creating the optical surface properties
    opticalSurfaceGadoxTb = new G4OpticalSurface("interfaceSurfaceGadoxTb");
    DefineOpticalSurface(opticalSurfaceGadoxTb);
}

void NSDetectorConstruction::DefineSiO2()
{
    SiO2 = new G4Material("SiO2", 2.65*g/cm3, 2);
    SiO2->AddElement(Si, 1);
    SiO2->AddElement(O, 2);

    rindexSiO2 = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexSiO2.dat", rindexSiO2);
    
    absLengthSiO2 = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffSiO2.dat", absLengthSiO2);
    
    G4MaterialPropertiesTable *mptSiO2 = DefineNonScintillatingMaterial(SiO2, 2, rindexSiO2, absLengthSiO2);
    
    // Creating the optical surface properties
    opticalSurfaceSiO2 = new G4OpticalSurface("opticalSurfaceSiO2");
    DefineOpticalSurface(opticalSurfaceSiO2);
}

void NSDetectorConstruction::DefineTiO2()
{
    TiO2 = new G4Material("TiO2", 4.23*g/cm3, 2);
    TiO2->AddElement(Ti, 1);
    TiO2->AddElement(O, 2);

    rindexTiO2 = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexTiO2.dat", rindexTiO2);
    
    absLengthTiO2 = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffTiO2.dat", absLengthTiO2);
    
    G4MaterialPropertiesTable *mptTiO2 = DefineNonScintillatingMaterial(TiO2, 2, rindexTiO2, absLengthTiO2);
    
    // Creating the optical surface properties
    opticalSurfaceTiO2 = new G4OpticalSurface("opticalSurfaceTiO2");
    DefineOpticalSurface(opticalSurfaceTiO2);
}

void NSDetectorConstruction::DefineAMBioMedia()
{
    // Teeth
    AMTeeth = new G4Material("AMTeeth", 2.75*g/cm3, 14);
    AMTeeth->AddElement(H, 0.022);
    AMTeeth->AddElement(C, 0.095);
    AMTeeth->AddElement(N, 0.029);
    AMTeeth->AddElement(O, 0.421);
    AMTeeth->AddElement(Na, 0.);
    AMTeeth->AddElement(Mg, 0.007);
    AMTeeth->AddElement(P, 0.137);
    AMTeeth->AddElement(S, 0.);
    AMTeeth->AddElement(Cl, 0.);
    AMTeeth->AddElement(K, 0.);
    AMTeeth->AddElement(Ca, 0.289);
    AMTeeth->AddElement(Fe, 0.);
    AMTeeth->AddElement(I, 0.);
    AMTeeth->AddElement(Gd, 0.);

    rindexAMTeeth = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMTeeth);
    
    absLengthAMTeeth = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMTeeth.dat", absLengthAMTeeth);
    
    G4MaterialPropertiesTable *mptAMTeeth = DefineNonScintillatingMaterial(AMTeeth, 14, rindexAMTeeth, absLengthAMTeeth);
    
    // MineralBone
    AMMineralBone = new G4Material("AMMineralBone", 1.92*g/cm3, 14);
    AMMineralBone->AddElement(H, 0.036);
    AMMineralBone->AddElement(C, 0.159);
    AMMineralBone->AddElement(N, 0.042);
    AMMineralBone->AddElement(O, 0.448);
    AMMineralBone->AddElement(Na, 0.003);
    AMMineralBone->AddElement(Mg, 0.002);
    AMMineralBone->AddElement(P, 0.094);
    AMMineralBone->AddElement(S, 0.003);
    AMMineralBone->AddElement(Cl, 0.);
    AMMineralBone->AddElement(K, 0.);
    AMMineralBone->AddElement(Ca, 0.213);
    AMMineralBone->AddElement(Fe, 0.);
    AMMineralBone->AddElement(I, 0.);
    AMMineralBone->AddElement(Gd, 0.);

    rindexAMMineralBone = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMMineralBone);
    
    absLengthAMMineralBone = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMMineralBone.dat", absLengthAMMineralBone);
    
    G4MaterialPropertiesTable *mptAMMineralBone = DefineNonScintillatingMaterial(AMMineralBone, 14, rindexAMMineralBone, absLengthAMMineralBone);
    
    // HumeriUpper
    AMHumeriUpper = new G4Material("AMHumeriUpper", 1.205*g/cm3, 14);
    AMHumeriUpper->AddElement(H, 0.085);
    AMHumeriUpper->AddElement(C, 0.288);
    AMHumeriUpper->AddElement(N, 0.026);
    AMHumeriUpper->AddElement(O, 0.498);
    AMHumeriUpper->AddElement(Na, 0.002);
    AMHumeriUpper->AddElement(Mg, 0.001);
    AMHumeriUpper->AddElement(P, 0.033);
    AMHumeriUpper->AddElement(S, 0.004);
    AMHumeriUpper->AddElement(Cl, 0.002);
    AMHumeriUpper->AddElement(K, 0.);
    AMHumeriUpper->AddElement(Ca, 0.061);
    AMHumeriUpper->AddElement(Fe, 0.);
    AMHumeriUpper->AddElement(I, 0.);
    AMHumeriUpper->AddElement(Gd, 0.);

    rindexAMHumeriUpper = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMHumeriUpper);
    
    absLengthAMHumeriUpper = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMHumeriUpper.dat", absLengthAMHumeriUpper);
    
    G4MaterialPropertiesTable *mptAMHumeriUpper = DefineNonScintillatingMaterial(AMHumeriUpper, 14, rindexAMHumeriUpper, absLengthAMHumeriUpper);
    
    // HumeriLower
    AMHumeriLower = new G4Material("AMHumeriLower", 1.108*g/cm3, 14);
    AMHumeriLower->AddElement(H, 0.097);
    AMHumeriLower->AddElement(C, 0.439);
    AMHumeriLower->AddElement(N, 0.017);
    AMHumeriLower->AddElement(O, 0.381);
    AMHumeriLower->AddElement(Na, 0.002);
    AMHumeriLower->AddElement(Mg, 0.);
    AMHumeriLower->AddElement(P, 0.021);
    AMHumeriLower->AddElement(S, 0.003);
    AMHumeriLower->AddElement(Cl, 0.001);
    AMHumeriLower->AddElement(K, 0.);
    AMHumeriLower->AddElement(Ca, 0.039);
    AMHumeriLower->AddElement(Fe, 0.);
    AMHumeriLower->AddElement(I, 0.);
    AMHumeriLower->AddElement(Gd, 0.);

    rindexAMHumeriLower = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMHumeriLower);
    
    absLengthAMHumeriLower = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMHumeriLower.dat", absLengthAMHumeriLower);
    
    G4MaterialPropertiesTable *mptAMHumeriLower = DefineNonScintillatingMaterial(AMHumeriLower, 14, rindexAMHumeriLower, absLengthAMHumeriLower);
    
    // LowerArmBone
    AMLowerArmBone = new G4Material("AMLowerArmBone", 1.108*g/cm3, 14);
    AMLowerArmBone->AddElement(H, 0.097);
    AMLowerArmBone->AddElement(C, 0.439);
    AMLowerArmBone->AddElement(N, 0.017);
    AMLowerArmBone->AddElement(O, 0.381);
    AMLowerArmBone->AddElement(Na, 0.002);
    AMLowerArmBone->AddElement(Mg, 0.);
    AMLowerArmBone->AddElement(P, 0.021);
    AMLowerArmBone->AddElement(S, 0.003);
    AMLowerArmBone->AddElement(Cl, 0.001);
    AMLowerArmBone->AddElement(K, 0.);
    AMLowerArmBone->AddElement(Ca, 0.039);
    AMLowerArmBone->AddElement(Fe, 0.);
    AMLowerArmBone->AddElement(I, 0.);
    AMLowerArmBone->AddElement(Gd, 0.);

    rindexAMLowerArmBone = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMLowerArmBone);
    
    absLengthAMLowerArmBone = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMLowerArmBone.dat", absLengthAMLowerArmBone);
    
    G4MaterialPropertiesTable *mptAMLowerArmBone = DefineNonScintillatingMaterial(AMLowerArmBone, 14, rindexAMLowerArmBone, absLengthAMLowerArmBone);
    
    // HandBone
    AMHandBone = new G4Material("AMHandBone", 1.108*g/cm3, 14);
    AMHandBone->AddElement(H, 0.097);
    AMHandBone->AddElement(C, 0.439);
    AMHandBone->AddElement(N, 0.017);
    AMHandBone->AddElement(O, 0.381);
    AMHandBone->AddElement(Na, 0.002);
    AMHandBone->AddElement(Mg, 0.);
    AMHandBone->AddElement(P, 0.021);
    AMHandBone->AddElement(S, 0.003);
    AMHandBone->AddElement(Cl, 0.001);
    AMHandBone->AddElement(K, 0.);
    AMHandBone->AddElement(Ca, 0.039);
    AMHandBone->AddElement(Fe, 0.);
    AMHandBone->AddElement(I, 0.);
    AMHandBone->AddElement(Gd, 0.);

    rindexAMHandBone = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMHandBone);
    
    absLengthAMHandBone = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMHandBone.dat", absLengthAMHandBone);
    
    G4MaterialPropertiesTable *mptAMHandBone = DefineNonScintillatingMaterial(AMHandBone, 14, rindexAMHandBone, absLengthAMHandBone);
    
    // Clavicles
    AMClavicles = new G4Material("AMClavicles", 1.151*g/cm3, 14);
    AMClavicles->AddElement(H, 0.091);
    AMClavicles->AddElement(C, 0.348);
    AMClavicles->AddElement(N, 0.024);
    AMClavicles->AddElement(O, 0.457);
    AMClavicles->AddElement(Na, 0.002);
    AMClavicles->AddElement(Mg, 0.);
    AMClavicles->AddElement(P, 0.026);
    AMClavicles->AddElement(S, 0.003);
    AMClavicles->AddElement(Cl, 0.001);
    AMClavicles->AddElement(K, 0.);
    AMClavicles->AddElement(Ca, 0.048);
    AMClavicles->AddElement(Fe, 0.);
    AMClavicles->AddElement(I, 0.);
    AMClavicles->AddElement(Gd, 0.);

    rindexAMClavicles = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMClavicles);
    
    absLengthAMClavicles = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMClavicles.dat", absLengthAMClavicles);
    
    G4MaterialPropertiesTable *mptAMClavicles = DefineNonScintillatingMaterial(AMClavicles, 14, rindexAMClavicles, absLengthAMClavicles);
    
    // Cranium
    AMCranium = new G4Material("AMCranium", 1.157*g/cm3, 14);
    AMCranium->AddElement(H, 0.09);
    AMCranium->AddElement(C, 0.335);
    AMCranium->AddElement(N, 0.025);
    AMCranium->AddElement(O, 0.467);
    AMCranium->AddElement(Na, 0.002);
    AMCranium->AddElement(Mg, 0.);
    AMCranium->AddElement(P, 0.026);
    AMCranium->AddElement(S, 0.003);
    AMCranium->AddElement(Cl, 0.002);
    AMCranium->AddElement(K, 0.001);
    AMCranium->AddElement(Ca, 0.049);
    AMCranium->AddElement(Fe, 0.);
    AMCranium->AddElement(I, 0.);
    AMCranium->AddElement(Gd, 0.);

    rindexAMCranium = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMCranium);
    
    absLengthAMCranium = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMCranium.dat", absLengthAMCranium);
    
    G4MaterialPropertiesTable *mptAMCranium = DefineNonScintillatingMaterial(AMCranium, 14, rindexAMCranium, absLengthAMCranium);
    
    // FemoraUpper
    AMFemoraUpper = new G4Material("AMFemoraUpper", 1.124*g/cm3, 14);
    AMFemoraUpper->AddElement(H, 0.094);
    AMFemoraUpper->AddElement(C, 0.385);
    AMFemoraUpper->AddElement(N, 0.022);
    AMFemoraUpper->AddElement(O, 0.43);
    AMFemoraUpper->AddElement(Na, 0.002);
    AMFemoraUpper->AddElement(Mg, 0.);
    AMFemoraUpper->AddElement(P, 0.022);
    AMFemoraUpper->AddElement(S, 0.003);
    AMFemoraUpper->AddElement(Cl, 0.001);
    AMFemoraUpper->AddElement(K, 0.);
    AMFemoraUpper->AddElement(Ca, 0.041);
    AMFemoraUpper->AddElement(Fe, 0.);
    AMFemoraUpper->AddElement(I, 0.);
    AMFemoraUpper->AddElement(Gd, 0.);

    rindexAMFemoraUpper = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMFemoraUpper);
    
    absLengthAMFemoraUpper = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMFemoraUpper.dat", absLengthAMFemoraUpper);
    
    G4MaterialPropertiesTable *mptAMFemoraUpper = DefineNonScintillatingMaterial(AMFemoraUpper, 14, rindexAMFemoraUpper, absLengthAMFemoraUpper);
    
    // FemoraLower
    AMFemoraLower = new G4Material("AMFemoraLower", 1.108*g/cm3, 14);
    AMFemoraLower->AddElement(H, 0.097);
    AMFemoraLower->AddElement(C, 0.439);
    AMFemoraLower->AddElement(N, 0.017);
    AMFemoraLower->AddElement(O, 0.381);
    AMFemoraLower->AddElement(Na, 0.002);
    AMFemoraLower->AddElement(Mg, 0.);
    AMFemoraLower->AddElement(P, 0.021);
    AMFemoraLower->AddElement(S, 0.003);
    AMFemoraLower->AddElement(Cl, 0.001);
    AMFemoraLower->AddElement(K, 0.);
    AMFemoraLower->AddElement(Ca, 0.039);
    AMFemoraLower->AddElement(Fe, 0.);
    AMFemoraLower->AddElement(I, 0.);
    AMFemoraLower->AddElement(Gd, 0.);

    rindexAMFemoraLower = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMFemoraLower);
    
    absLengthAMFemoraLower = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMFemoraLower.dat", absLengthAMFemoraLower);
    
    G4MaterialPropertiesTable *mptAMFemoraLower = DefineNonScintillatingMaterial(AMFemoraLower, 14, rindexAMFemoraLower, absLengthAMFemoraLower);
    
    // LowerLeg
    AMLowerLeg = new G4Material("AMLowerLeg", 1.108*g/cm3, 14);
    AMLowerLeg->AddElement(H, 0.097);
    AMLowerLeg->AddElement(C, 0.439);
    AMLowerLeg->AddElement(N, 0.017);
    AMLowerLeg->AddElement(O, 0.381);
    AMLowerLeg->AddElement(Na, 0.002);
    AMLowerLeg->AddElement(Mg, 0.);
    AMLowerLeg->AddElement(P, 0.021);
    AMLowerLeg->AddElement(S, 0.003);
    AMLowerLeg->AddElement(Cl, 0.001);
    AMLowerLeg->AddElement(K, 0.);
    AMLowerLeg->AddElement(Ca, 0.039);
    AMLowerLeg->AddElement(Fe, 0.);
    AMLowerLeg->AddElement(I, 0.);
    AMLowerLeg->AddElement(Gd, 0.);

    rindexAMLowerLeg = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMLowerLeg);
    
    absLengthAMLowerLeg = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMLowerLeg.dat", absLengthAMLowerLeg);
    
    G4MaterialPropertiesTable *mptAMLowerLeg = DefineNonScintillatingMaterial(AMLowerLeg, 14, rindexAMLowerLeg, absLengthAMLowerLeg);
    
    // Foot
    AMFoot = new G4Material("AMFoot", 1.108*g/cm3, 14);
    AMFoot->AddElement(H, 0.097);
    AMFoot->AddElement(C, 0.439);
    AMFoot->AddElement(N, 0.017);
    AMFoot->AddElement(O, 0.381);
    AMFoot->AddElement(Na, 0.002);
    AMFoot->AddElement(Mg, 0.);
    AMFoot->AddElement(P, 0.021);
    AMFoot->AddElement(S, 0.003);
    AMFoot->AddElement(Cl, 0.001);
    AMFoot->AddElement(K, 0.);
    AMFoot->AddElement(Ca, 0.039);
    AMFoot->AddElement(Fe, 0.);
    AMFoot->AddElement(I, 0.);
    AMFoot->AddElement(Gd, 0.);

    rindexAMFoot = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMFoot);
    
    absLengthAMFoot = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMFoot.dat", absLengthAMFoot);
    
    G4MaterialPropertiesTable *mptAMFoot = DefineNonScintillatingMaterial(AMFoot, 14, rindexAMFoot, absLengthAMFoot);
    
    // Mandible
    AMMandible = new G4Material("AMMandible", 1.228*g/cm3, 14);
    AMMandible->AddElement(H, 0.083);
    AMMandible->AddElement(C, 0.266);
    AMMandible->AddElement(N, 0.027);
    AMMandible->AddElement(O, 0.511);
    AMMandible->AddElement(Na, 0.003);
    AMMandible->AddElement(Mg, 0.001);
    AMMandible->AddElement(P, 0.036);
    AMMandible->AddElement(S, 0.004);
    AMMandible->AddElement(Cl, 0.002);
    AMMandible->AddElement(K, 0.);
    AMMandible->AddElement(Ca, 0.067);
    AMMandible->AddElement(Fe, 0.);
    AMMandible->AddElement(I, 0.);
    AMMandible->AddElement(Gd, 0.);

    rindexAMMandible = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMMandible);
    
    absLengthAMMandible = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMMandible.dat", absLengthAMMandible);
    
    G4MaterialPropertiesTable *mptAMMandible = DefineNonScintillatingMaterial(AMMandible, 14, rindexAMMandible, absLengthAMMandible);
    
    // Pelvis
    AMPelvis = new G4Material("AMPelvis", 1.123*g/cm3, 14);
    AMPelvis->AddElement(H, 0.094);
    AMPelvis->AddElement(C, 0.36);
    AMPelvis->AddElement(N, 0.025);
    AMPelvis->AddElement(O, 0.454);
    AMPelvis->AddElement(Na, 0.002);
    AMPelvis->AddElement(Mg, 0.);
    AMPelvis->AddElement(P, 0.021);
    AMPelvis->AddElement(S, 0.003);
    AMPelvis->AddElement(Cl, 0.002);
    AMPelvis->AddElement(K, 0.001);
    AMPelvis->AddElement(Ca, 0.038);
    AMPelvis->AddElement(Fe, 0.);
    AMPelvis->AddElement(I, 0.);
    AMPelvis->AddElement(Gd, 0.);

    rindexAMPelvis = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMPelvis);
    
    absLengthAMPelvis = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMPelvis.dat", absLengthAMPelvis);
    
    G4MaterialPropertiesTable *mptAMPelvis = DefineNonScintillatingMaterial(AMPelvis, 14, rindexAMPelvis, absLengthAMPelvis);
    
    // Ribs
    AMRibs = new G4Material("AMRibs", 1.165*g/cm3, 14);
    AMRibs->AddElement(H, 0.089);
    AMRibs->AddElement(C, 0.292);
    AMRibs->AddElement(N, 0.029);
    AMRibs->AddElement(O, 0.507);
    AMRibs->AddElement(Na, 0.002);
    AMRibs->AddElement(Mg, 0.);
    AMRibs->AddElement(P, 0.026);
    AMRibs->AddElement(S, 0.004);
    AMRibs->AddElement(Cl, 0.002);
    AMRibs->AddElement(K, 0.001);
    AMRibs->AddElement(Ca, 0.048);
    AMRibs->AddElement(Fe, 0.);
    AMRibs->AddElement(I, 0.);
    AMRibs->AddElement(Gd, 0.);

    rindexAMRibs = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMRibs);
    
    absLengthAMRibs = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMRibs.dat", absLengthAMRibs);
    
    G4MaterialPropertiesTable *mptAMRibs = DefineNonScintillatingMaterial(AMRibs, 14, rindexAMRibs, absLengthAMRibs);
    
    // Scapulae
    AMScapulae = new G4Material("AMScapulae", 1.183*g/cm3, 14);
    AMScapulae->AddElement(H, 0.087);
    AMScapulae->AddElement(C, 0.309);
    AMScapulae->AddElement(N, 0.026);
    AMScapulae->AddElement(O, 0.483);
    AMScapulae->AddElement(Na, 0.002);
    AMScapulae->AddElement(Mg, 0.001);
    AMScapulae->AddElement(P, 0.03);
    AMScapulae->AddElement(S, 0.004);
    AMScapulae->AddElement(Cl, 0.002);
    AMScapulae->AddElement(K, 0.);
    AMScapulae->AddElement(Ca, 0.056);
    AMScapulae->AddElement(Fe, 0.);
    AMScapulae->AddElement(I, 0.);
    AMScapulae->AddElement(Gd, 0.);

    rindexAMScapulae = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMScapulae);
    
    absLengthAMScapulae = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMScapulae.dat", absLengthAMScapulae);
    
    G4MaterialPropertiesTable *mptAMScapulae = DefineNonScintillatingMaterial(AMScapulae, 14, rindexAMScapulae, absLengthAMScapulae);
    
    // CervicalSpine
    AMCervicalSpine = new G4Material("AMCervicalSpine", 1.05*g/cm3, 14);
    AMCervicalSpine->AddElement(H, 0.103);
    AMCervicalSpine->AddElement(C, 0.4);
    AMCervicalSpine->AddElement(N, 0.027);
    AMCervicalSpine->AddElement(O, 0.444);
    AMCervicalSpine->AddElement(Na, 0.001);
    AMCervicalSpine->AddElement(Mg, 0.);
    AMCervicalSpine->AddElement(P, 0.007);
    AMCervicalSpine->AddElement(S, 0.002);
    AMCervicalSpine->AddElement(Cl, 0.002);
    AMCervicalSpine->AddElement(K, 0.001);
    AMCervicalSpine->AddElement(Ca, 0.012);
    AMCervicalSpine->AddElement(Fe, 0.001);
    AMCervicalSpine->AddElement(I, 0.);
    AMCervicalSpine->AddElement(Gd, 0.);

    rindexAMCervicalSpine = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMCervicalSpine);
    
    absLengthAMCervicalSpine = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMCervicalSpine.dat", absLengthAMCervicalSpine);
    
    G4MaterialPropertiesTable *mptAMCervicalSpine = DefineNonScintillatingMaterial(AMCervicalSpine, 14, rindexAMCervicalSpine, absLengthAMCervicalSpine);
    
    // ThoracicSpine
    AMThoracicSpine = new G4Material("AMThoracicSpine", 1.074*g/cm3, 14);
    AMThoracicSpine->AddElement(H, 0.099);
    AMThoracicSpine->AddElement(C, 0.376);
    AMThoracicSpine->AddElement(N, 0.027);
    AMThoracicSpine->AddElement(O, 0.459);
    AMThoracicSpine->AddElement(Na, 0.001);
    AMThoracicSpine->AddElement(Mg, 0.);
    AMThoracicSpine->AddElement(P, 0.012);
    AMThoracicSpine->AddElement(S, 0.002);
    AMThoracicSpine->AddElement(Cl, 0.002);
    AMThoracicSpine->AddElement(K, 0.001);
    AMThoracicSpine->AddElement(Ca, 0.02);
    AMThoracicSpine->AddElement(Fe, 0.001);
    AMThoracicSpine->AddElement(I, 0.);
    AMThoracicSpine->AddElement(Gd, 0.);

    rindexAMThoracicSpine = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMThoracicSpine);
    
    absLengthAMThoracicSpine = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMThoracicSpine.dat", absLengthAMThoracicSpine);
    
    G4MaterialPropertiesTable *mptAMThoracicSpine = DefineNonScintillatingMaterial(AMThoracicSpine, 14, rindexAMThoracicSpine, absLengthAMThoracicSpine);
    
    // LumbarSpine
    AMLumbarSpine = new G4Material("AMLumbarSpine", 1.112*g/cm3, 14);
    AMLumbarSpine->AddElement(H, 0.095);
    AMLumbarSpine->AddElement(C, 0.34);
    AMLumbarSpine->AddElement(N, 0.028);
    AMLumbarSpine->AddElement(O, 0.48);
    AMLumbarSpine->AddElement(Na, 0.001);
    AMLumbarSpine->AddElement(Mg, 0.);
    AMLumbarSpine->AddElement(P, 0.018);
    AMLumbarSpine->AddElement(S, 0.003);
    AMLumbarSpine->AddElement(Cl, 0.002);
    AMLumbarSpine->AddElement(K, 0.001);
    AMLumbarSpine->AddElement(Ca, 0.032);
    AMLumbarSpine->AddElement(Fe, 0.);
    AMLumbarSpine->AddElement(I, 0.);
    AMLumbarSpine->AddElement(Gd, 0.);

    rindexAMLumbarSpine = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMLumbarSpine);
    
    absLengthAMLumbarSpine = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMLumbarSpine.dat", absLengthAMLumbarSpine);
    
    G4MaterialPropertiesTable *mptAMLumbarSpine = DefineNonScintillatingMaterial(AMLumbarSpine, 14, rindexAMLumbarSpine, absLengthAMLumbarSpine);
    
    // Sacrum
    AMSacrum = new G4Material("AMSacrum", 1.031*g/cm3, 14);
    AMSacrum->AddElement(H, 0.105);
    AMSacrum->AddElement(C, 0.419);
    AMSacrum->AddElement(N, 0.027);
    AMSacrum->AddElement(O, 0.432);
    AMSacrum->AddElement(Na, 0.001);
    AMSacrum->AddElement(Mg, 0.);
    AMSacrum->AddElement(P, 0.004);
    AMSacrum->AddElement(S, 0.002);
    AMSacrum->AddElement(Cl, 0.002);
    AMSacrum->AddElement(K, 0.001);
    AMSacrum->AddElement(Ca, 0.006);
    AMSacrum->AddElement(Fe, 0.001);
    AMSacrum->AddElement(I, 0.);
    AMSacrum->AddElement(Gd, 0.);

    rindexAMSacrum = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMSacrum);
    
    absLengthAMSacrum = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMSacrum.dat", absLengthAMSacrum);
    
    G4MaterialPropertiesTable *mptAMSacrum = DefineNonScintillatingMaterial(AMSacrum, 14, rindexAMSacrum, absLengthAMSacrum);
    
    // Sternum
    AMSternum = new G4Material("AMSternum", 1.041*g/cm3, 14);
    AMSternum->AddElement(H, 0.104);
    AMSternum->AddElement(C, 0.409);
    AMSternum->AddElement(N, 0.027);
    AMSternum->AddElement(O, 0.438);
    AMSternum->AddElement(Na, 0.001);
    AMSternum->AddElement(Mg, 0.);
    AMSternum->AddElement(P, 0.006);
    AMSternum->AddElement(S, 0.002);
    AMSternum->AddElement(Cl, 0.002);
    AMSternum->AddElement(K, 0.001);
    AMSternum->AddElement(Ca, 0.009);
    AMSternum->AddElement(Fe, 0.001);
    AMSternum->AddElement(I, 0.);
    AMSternum->AddElement(Gd, 0.);

    rindexAMSternum = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMSternum);
    
    absLengthAMSternum = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMSternum.dat", absLengthAMSternum);
    
    G4MaterialPropertiesTable *mptAMSternum = DefineNonScintillatingMaterial(AMSternum, 14, rindexAMSternum, absLengthAMSternum);
    
    // HumeriFemoraUpperMedullaryCavity
    AMHumeriFemoraUpperMedullaryCavity = new G4Material("AMHumeriFemoraUpperMedullaryCavity", 0.98*g/cm3, 14);
    AMHumeriFemoraUpperMedullaryCavity->AddElement(H, 0.115);
    AMHumeriFemoraUpperMedullaryCavity->AddElement(C, 0.636);
    AMHumeriFemoraUpperMedullaryCavity->AddElement(N, 0.007);
    AMHumeriFemoraUpperMedullaryCavity->AddElement(O, 0.239);
    AMHumeriFemoraUpperMedullaryCavity->AddElement(Na, 0.001);
    AMHumeriFemoraUpperMedullaryCavity->AddElement(Mg, 0.);
    AMHumeriFemoraUpperMedullaryCavity->AddElement(P, 0.);
    AMHumeriFemoraUpperMedullaryCavity->AddElement(S, 0.001);
    AMHumeriFemoraUpperMedullaryCavity->AddElement(Cl, 0.001);
    AMHumeriFemoraUpperMedullaryCavity->AddElement(K, 0.);
    AMHumeriFemoraUpperMedullaryCavity->AddElement(Ca, 0.);
    AMHumeriFemoraUpperMedullaryCavity->AddElement(Fe, 0.);
    AMHumeriFemoraUpperMedullaryCavity->AddElement(I, 0.);
    AMHumeriFemoraUpperMedullaryCavity->AddElement(Gd, 0.);

    rindexAMHumeriFemoraUpperMedullaryCavity = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMHumeriFemoraUpperMedullaryCavity);
    
    absLengthAMHumeriFemoraUpperMedullaryCavity = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMHumeriFemoraUpperMedullaryCavity.dat", absLengthAMHumeriFemoraUpperMedullaryCavity);
    
    G4MaterialPropertiesTable *mptAMHumeriFemoraUpperMedullaryCavity = DefineNonScintillatingMaterial(AMHumeriFemoraUpperMedullaryCavity, 14, rindexAMHumeriFemoraUpperMedullaryCavity, absLengthAMHumeriFemoraUpperMedullaryCavity);
    
    // HumeriFemoraLowerMedullaryCavity
    AMHumeriFemoraLowerMedullaryCavity = new G4Material("AMHumeriFemoraLowerMedullaryCavity", 0.98*g/cm3, 14);
    AMHumeriFemoraLowerMedullaryCavity->AddElement(H, 0.115);
    AMHumeriFemoraLowerMedullaryCavity->AddElement(C, 0.636);
    AMHumeriFemoraLowerMedullaryCavity->AddElement(N, 0.007);
    AMHumeriFemoraLowerMedullaryCavity->AddElement(O, 0.239);
    AMHumeriFemoraLowerMedullaryCavity->AddElement(Na, 0.001);
    AMHumeriFemoraLowerMedullaryCavity->AddElement(Mg, 0.);
    AMHumeriFemoraLowerMedullaryCavity->AddElement(P, 0.);
    AMHumeriFemoraLowerMedullaryCavity->AddElement(S, 0.001);
    AMHumeriFemoraLowerMedullaryCavity->AddElement(Cl, 0.001);
    AMHumeriFemoraLowerMedullaryCavity->AddElement(K, 0.);
    AMHumeriFemoraLowerMedullaryCavity->AddElement(Ca, 0.);
    AMHumeriFemoraLowerMedullaryCavity->AddElement(Fe, 0.);
    AMHumeriFemoraLowerMedullaryCavity->AddElement(I, 0.);
    AMHumeriFemoraLowerMedullaryCavity->AddElement(Gd, 0.);

    rindexAMHumeriFemoraLowerMedullaryCavity = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMHumeriFemoraLowerMedullaryCavity);
    
    absLengthAMHumeriFemoraLowerMedullaryCavity = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMHumeriFemoraLowerMedullaryCavity.dat", absLengthAMHumeriFemoraLowerMedullaryCavity);
    
    G4MaterialPropertiesTable *mptAMHumeriFemoraLowerMedullaryCavity = DefineNonScintillatingMaterial(AMHumeriFemoraLowerMedullaryCavity, 14, rindexAMHumeriFemoraLowerMedullaryCavity, absLengthAMHumeriFemoraLowerMedullaryCavity);
    
    // LowerArmMedullaryCavity
    AMLowerArmMedullaryCavity = new G4Material("AMLowerArmMedullaryCavity", 0.98*g/cm3, 14);
    AMLowerArmMedullaryCavity->AddElement(H, 0.115);
    AMLowerArmMedullaryCavity->AddElement(C, 0.636);
    AMLowerArmMedullaryCavity->AddElement(N, 0.007);
    AMLowerArmMedullaryCavity->AddElement(O, 0.239);
    AMLowerArmMedullaryCavity->AddElement(Na, 0.001);
    AMLowerArmMedullaryCavity->AddElement(Mg, 0.);
    AMLowerArmMedullaryCavity->AddElement(P, 0.);
    AMLowerArmMedullaryCavity->AddElement(S, 0.001);
    AMLowerArmMedullaryCavity->AddElement(Cl, 0.001);
    AMLowerArmMedullaryCavity->AddElement(K, 0.);
    AMLowerArmMedullaryCavity->AddElement(Ca, 0.);
    AMLowerArmMedullaryCavity->AddElement(Fe, 0.);
    AMLowerArmMedullaryCavity->AddElement(I, 0.);
    AMLowerArmMedullaryCavity->AddElement(Gd, 0.);

    rindexAMLowerArmMedullaryCavity = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMLowerArmMedullaryCavity);
    
    absLengthAMLowerArmMedullaryCavity = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMLowerArmMedullaryCavity.dat", absLengthAMLowerArmMedullaryCavity);
    
    G4MaterialPropertiesTable *mptAMLowerArmMedullaryCavity = DefineNonScintillatingMaterial(AMLowerArmMedullaryCavity, 14, rindexAMLowerArmMedullaryCavity, absLengthAMLowerArmMedullaryCavity);
    
    // LowerLegMedullaryCavity
    AMLowerLegMedullaryCavity = new G4Material("AMLowerLegMedullaryCavity", 0.98*g/cm3, 14);
    AMLowerLegMedullaryCavity->AddElement(H, 0.115);
    AMLowerLegMedullaryCavity->AddElement(C, 0.636);
    AMLowerLegMedullaryCavity->AddElement(N, 0.007);
    AMLowerLegMedullaryCavity->AddElement(O, 0.239);
    AMLowerLegMedullaryCavity->AddElement(Na, 0.001);
    AMLowerLegMedullaryCavity->AddElement(Mg, 0.);
    AMLowerLegMedullaryCavity->AddElement(P, 0.);
    AMLowerLegMedullaryCavity->AddElement(S, 0.001);
    AMLowerLegMedullaryCavity->AddElement(Cl, 0.001);
    AMLowerLegMedullaryCavity->AddElement(K, 0.);
    AMLowerLegMedullaryCavity->AddElement(Ca, 0.);
    AMLowerLegMedullaryCavity->AddElement(Fe, 0.);
    AMLowerLegMedullaryCavity->AddElement(I, 0.);
    AMLowerLegMedullaryCavity->AddElement(Gd, 0.);

    rindexAMLowerLegMedullaryCavity = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMLowerLegMedullaryCavity);
    
    absLengthAMLowerLegMedullaryCavity = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMLowerLegMedullaryCavity.dat", absLengthAMLowerLegMedullaryCavity);
    
    G4MaterialPropertiesTable *mptAMLowerLegMedullaryCavity = DefineNonScintillatingMaterial(AMLowerLegMedullaryCavity, 14, rindexAMLowerLegMedullaryCavity, absLengthAMLowerLegMedullaryCavity);
    
    // Cartilage
    AMCartilage = new G4Material("AMCartilage", 1.1*g/cm3, 14);
    AMCartilage->AddElement(H, 0.096);
    AMCartilage->AddElement(C, 0.099);
    AMCartilage->AddElement(N, 0.022);
    AMCartilage->AddElement(O, 0.744);
    AMCartilage->AddElement(Na, 0.005);
    AMCartilage->AddElement(Mg, 0.);
    AMCartilage->AddElement(P, 0.022);
    AMCartilage->AddElement(S, 0.009);
    AMCartilage->AddElement(Cl, 0.003);
    AMCartilage->AddElement(K, 0.);
    AMCartilage->AddElement(Ca, 0.);
    AMCartilage->AddElement(Fe, 0.);
    AMCartilage->AddElement(I, 0.);
    AMCartilage->AddElement(Gd, 0.);

    rindexAMCartilage = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMCartilage);
    
    absLengthAMCartilage = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMCartilage.dat", absLengthAMCartilage);
    
    G4MaterialPropertiesTable *mptAMCartilage = DefineNonScintillatingMaterial(AMCartilage, 14, rindexAMCartilage, absLengthAMCartilage);
    
    // Skin
    AMSkin = new G4Material("AMSkin", 1.09*g/cm3, 14);
    AMSkin->AddElement(H, 0.1);
    AMSkin->AddElement(C, 0.199);
    AMSkin->AddElement(N, 0.042);
    AMSkin->AddElement(O, 0.65);
    AMSkin->AddElement(Na, 0.002);
    AMSkin->AddElement(Mg, 0.);
    AMSkin->AddElement(P, 0.001);
    AMSkin->AddElement(S, 0.002);
    AMSkin->AddElement(Cl, 0.003);
    AMSkin->AddElement(K, 0.001);
    AMSkin->AddElement(Ca, 0.);
    AMSkin->AddElement(Fe, 0.);
    AMSkin->AddElement(I, 0.);
    AMSkin->AddElement(Gd, 0.);

    rindexAMSkin = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMSkin);
    
    absLengthAMSkin = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMSkin.dat", absLengthAMSkin);
    
    G4MaterialPropertiesTable *mptAMSkin = DefineNonScintillatingMaterial(AMSkin, 14, rindexAMSkin, absLengthAMSkin);
    
    // Blood
    AMBlood = new G4Material("AMBlood", 1.06*g/cm3, 14);
    AMBlood->AddElement(H, 0.102);
    AMBlood->AddElement(C, 0.11);
    AMBlood->AddElement(N, 0.033);
    AMBlood->AddElement(O, 0.745);
    AMBlood->AddElement(Na, 0.001);
    AMBlood->AddElement(Mg, 0.);
    AMBlood->AddElement(P, 0.001);
    AMBlood->AddElement(S, 0.002);
    AMBlood->AddElement(Cl, 0.003);
    AMBlood->AddElement(K, 0.002);
    AMBlood->AddElement(Ca, 0.);
    AMBlood->AddElement(Fe, 0.001);
    AMBlood->AddElement(I, 0.);
    AMBlood->AddElement(Gd, 0.);

    rindexAMBlood = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMBlood);
    
    absLengthAMBlood = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMBlood.dat", absLengthAMBlood);
    
    G4MaterialPropertiesTable *mptAMBlood = DefineNonScintillatingMaterial(AMBlood, 14, rindexAMBlood, absLengthAMBlood);
    
    // Muscle
    AMMuscle = new G4Material("AMMuscle", 1.05*g/cm3, 14);
    AMMuscle->AddElement(H, 0.102);
    AMMuscle->AddElement(C, 0.142);
    AMMuscle->AddElement(N, 0.034);
    AMMuscle->AddElement(O, 0.711);
    AMMuscle->AddElement(Na, 0.001);
    AMMuscle->AddElement(Mg, 0.);
    AMMuscle->AddElement(P, 0.002);
    AMMuscle->AddElement(S, 0.003);
    AMMuscle->AddElement(Cl, 0.001);
    AMMuscle->AddElement(K, 0.004);
    AMMuscle->AddElement(Ca, 0.);
    AMMuscle->AddElement(Fe, 0.);
    AMMuscle->AddElement(I, 0.);
    AMMuscle->AddElement(Gd, 0.);

    rindexAMMuscle = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMMuscle);
    
    absLengthAMMuscle = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMMuscle.dat", absLengthAMMuscle);
    
    G4MaterialPropertiesTable *mptAMMuscle = DefineNonScintillatingMaterial(AMMuscle, 14, rindexAMMuscle, absLengthAMMuscle);
    
    // Liver
    AMLiver = new G4Material("AMLiver", 1.05*g/cm3, 14);
    AMLiver->AddElement(H, 0.102);
    AMLiver->AddElement(C, 0.13);
    AMLiver->AddElement(N, 0.031);
    AMLiver->AddElement(O, 0.725);
    AMLiver->AddElement(Na, 0.002);
    AMLiver->AddElement(Mg, 0.);
    AMLiver->AddElement(P, 0.002);
    AMLiver->AddElement(S, 0.003);
    AMLiver->AddElement(Cl, 0.002);
    AMLiver->AddElement(K, 0.003);
    AMLiver->AddElement(Ca, 0.);
    AMLiver->AddElement(Fe, 0.);
    AMLiver->AddElement(I, 0.);
    AMLiver->AddElement(Gd, 0.);

    rindexAMLiver = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMLiver);
    
    absLengthAMLiver = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMLiver.dat", absLengthAMLiver);
    
    G4MaterialPropertiesTable *mptAMLiver = DefineNonScintillatingMaterial(AMLiver, 14, rindexAMLiver, absLengthAMLiver);
    
    // Pancreas
    AMPancreas = new G4Material("AMPancreas", 1.05*g/cm3, 14);
    AMPancreas->AddElement(H, 0.105);
    AMPancreas->AddElement(C, 0.155);
    AMPancreas->AddElement(N, 0.025);
    AMPancreas->AddElement(O, 0.706);
    AMPancreas->AddElement(Na, 0.002);
    AMPancreas->AddElement(Mg, 0.);
    AMPancreas->AddElement(P, 0.002);
    AMPancreas->AddElement(S, 0.001);
    AMPancreas->AddElement(Cl, 0.002);
    AMPancreas->AddElement(K, 0.002);
    AMPancreas->AddElement(Ca, 0.);
    AMPancreas->AddElement(Fe, 0.);
    AMPancreas->AddElement(I, 0.);
    AMPancreas->AddElement(Gd, 0.);

    rindexAMPancreas = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMPancreas);
    
    absLengthAMPancreas = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMPancreas.dat", absLengthAMPancreas);
    
    G4MaterialPropertiesTable *mptAMPancreas = DefineNonScintillatingMaterial(AMPancreas, 14, rindexAMPancreas, absLengthAMPancreas);
    
    // Brain
    AMBrain = new G4Material("AMBrain", 1.05*g/cm3, 14);
    AMBrain->AddElement(H, 0.107);
    AMBrain->AddElement(C, 0.143);
    AMBrain->AddElement(N, 0.023);
    AMBrain->AddElement(O, 0.713);
    AMBrain->AddElement(Na, 0.002);
    AMBrain->AddElement(Mg, 0.);
    AMBrain->AddElement(P, 0.004);
    AMBrain->AddElement(S, 0.002);
    AMBrain->AddElement(Cl, 0.003);
    AMBrain->AddElement(K, 0.003);
    AMBrain->AddElement(Ca, 0.);
    AMBrain->AddElement(Fe, 0.);
    AMBrain->AddElement(I, 0.);
    AMBrain->AddElement(Gd, 0.);

    rindexAMBrain = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMBrain);
    
    absLengthAMBrain = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMBrain.dat", absLengthAMBrain);
    
    G4MaterialPropertiesTable *mptAMBrain = DefineNonScintillatingMaterial(AMBrain, 14, rindexAMBrain, absLengthAMBrain);
    
    // Heart
    AMHeart = new G4Material("AMHeart", 1.05*g/cm3, 14);
    AMHeart->AddElement(H, 0.104);
    AMHeart->AddElement(C, 0.138);
    AMHeart->AddElement(N, 0.029);
    AMHeart->AddElement(O, 0.719);
    AMHeart->AddElement(Na, 0.001);
    AMHeart->AddElement(Mg, 0.);
    AMHeart->AddElement(P, 0.002);
    AMHeart->AddElement(S, 0.002);
    AMHeart->AddElement(Cl, 0.002);
    AMHeart->AddElement(K, 0.003);
    AMHeart->AddElement(Ca, 0.);
    AMHeart->AddElement(Fe, 0.);
    AMHeart->AddElement(I, 0.);
    AMHeart->AddElement(Gd, 0.);

    rindexAMHeart = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMHeart);
    
    absLengthAMHeart = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMHeart.dat", absLengthAMHeart);
    
    G4MaterialPropertiesTable *mptAMHeart = DefineNonScintillatingMaterial(AMHeart, 14, rindexAMHeart, absLengthAMHeart);
    
    // Eyes
    AMEyes = new G4Material("AMEyes", 1.05*g/cm3, 14);
    AMEyes->AddElement(H, 0.097);
    AMEyes->AddElement(C, 0.181);
    AMEyes->AddElement(N, 0.053);
    AMEyes->AddElement(O, 0.663);
    AMEyes->AddElement(Na, 0.001);
    AMEyes->AddElement(Mg, 0.);
    AMEyes->AddElement(P, 0.001);
    AMEyes->AddElement(S, 0.003);
    AMEyes->AddElement(Cl, 0.001);
    AMEyes->AddElement(K, 0.);
    AMEyes->AddElement(Ca, 0.);
    AMEyes->AddElement(Fe, 0.);
    AMEyes->AddElement(I, 0.);
    AMEyes->AddElement(Gd, 0.);

    rindexAMEyes = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMEyes);
    
    absLengthAMEyes = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMEyes.dat", absLengthAMEyes);
    
    G4MaterialPropertiesTable *mptAMEyes = DefineNonScintillatingMaterial(AMEyes, 14, rindexAMEyes, absLengthAMEyes);
    
    // Kidneys
    AMKidneys = new G4Material("AMKidneys", 1.05*g/cm3, 14);
    AMKidneys->AddElement(H, 0.103);
    AMKidneys->AddElement(C, 0.124);
    AMKidneys->AddElement(N, 0.031);
    AMKidneys->AddElement(O, 0.731);
    AMKidneys->AddElement(Na, 0.002);
    AMKidneys->AddElement(Mg, 0.);
    AMKidneys->AddElement(P, 0.002);
    AMKidneys->AddElement(S, 0.002);
    AMKidneys->AddElement(Cl, 0.002);
    AMKidneys->AddElement(K, 0.002);
    AMKidneys->AddElement(Ca, 0.001);
    AMKidneys->AddElement(Fe, 0.);
    AMKidneys->AddElement(I, 0.);
    AMKidneys->AddElement(Gd, 0.);

    rindexAMKidneys = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMKidneys);
    
    absLengthAMKidneys = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMKidneys.dat", absLengthAMKidneys);
    
    G4MaterialPropertiesTable *mptAMKidneys = DefineNonScintillatingMaterial(AMKidneys, 14, rindexAMKidneys, absLengthAMKidneys);
    
    // Stomach
    AMStomach = new G4Material("AMStomach", 1.04*g/cm3, 14);
    AMStomach->AddElement(H, 0.105);
    AMStomach->AddElement(C, 0.114);
    AMStomach->AddElement(N, 0.025);
    AMStomach->AddElement(O, 0.75);
    AMStomach->AddElement(Na, 0.001);
    AMStomach->AddElement(Mg, 0.);
    AMStomach->AddElement(P, 0.001);
    AMStomach->AddElement(S, 0.001);
    AMStomach->AddElement(Cl, 0.002);
    AMStomach->AddElement(K, 0.001);
    AMStomach->AddElement(Ca, 0.);
    AMStomach->AddElement(Fe, 0.);
    AMStomach->AddElement(I, 0.);
    AMStomach->AddElement(Gd, 0.);

    rindexAMStomach = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMStomach);
    
    absLengthAMStomach = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMStomach.dat", absLengthAMStomach);
    
    G4MaterialPropertiesTable *mptAMStomach = DefineNonScintillatingMaterial(AMStomach, 14, rindexAMStomach, absLengthAMStomach);
    
    // SmallIntestine
    AMSmallIntestine = new G4Material("AMSmallIntestine", 1.04*g/cm3, 14);
    AMSmallIntestine->AddElement(H, 0.105);
    AMSmallIntestine->AddElement(C, 0.113);
    AMSmallIntestine->AddElement(N, 0.026);
    AMSmallIntestine->AddElement(O, 0.75);
    AMSmallIntestine->AddElement(Na, 0.001);
    AMSmallIntestine->AddElement(Mg, 0.);
    AMSmallIntestine->AddElement(P, 0.001);
    AMSmallIntestine->AddElement(S, 0.001);
    AMSmallIntestine->AddElement(Cl, 0.002);
    AMSmallIntestine->AddElement(K, 0.001);
    AMSmallIntestine->AddElement(Ca, 0.);
    AMSmallIntestine->AddElement(Fe, 0.);
    AMSmallIntestine->AddElement(I, 0.);
    AMSmallIntestine->AddElement(Gd, 0.);

    rindexAMSmallIntestine = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMSmallIntestine);
    
    absLengthAMSmallIntestine = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMSmallIntestine.dat", absLengthAMSmallIntestine);
    
    G4MaterialPropertiesTable *mptAMSmallIntestine = DefineNonScintillatingMaterial(AMSmallIntestine, 14, rindexAMSmallIntestine, absLengthAMSmallIntestine);
    
    // LargeIntestine
    AMLargeIntestine = new G4Material("AMLargeIntestine", 1.04*g/cm3, 14);
    AMLargeIntestine->AddElement(H, 0.105);
    AMLargeIntestine->AddElement(C, 0.113);
    AMLargeIntestine->AddElement(N, 0.026);
    AMLargeIntestine->AddElement(O, 0.75);
    AMLargeIntestine->AddElement(Na, 0.001);
    AMLargeIntestine->AddElement(Mg, 0.);
    AMLargeIntestine->AddElement(P, 0.001);
    AMLargeIntestine->AddElement(S, 0.001);
    AMLargeIntestine->AddElement(Cl, 0.002);
    AMLargeIntestine->AddElement(K, 0.001);
    AMLargeIntestine->AddElement(Ca, 0.);
    AMLargeIntestine->AddElement(Fe, 0.);
    AMLargeIntestine->AddElement(I, 0.);
    AMLargeIntestine->AddElement(Gd, 0.);

    rindexAMLargeIntestine = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMLargeIntestine);
    
    absLengthAMLargeIntestine = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMLargeIntestine.dat", absLengthAMLargeIntestine);
    
    G4MaterialPropertiesTable *mptAMLargeIntestine = DefineNonScintillatingMaterial(AMLargeIntestine, 14, rindexAMLargeIntestine, absLengthAMLargeIntestine);
    
    // Spleen
    AMSpleen = new G4Material("AMSpleen", 1.04*g/cm3, 14);
    AMSpleen->AddElement(H, 0.102);
    AMSpleen->AddElement(C, 0.111);
    AMSpleen->AddElement(N, 0.033);
    AMSpleen->AddElement(O, 0.743);
    AMSpleen->AddElement(Na, 0.001);
    AMSpleen->AddElement(Mg, 0.);
    AMSpleen->AddElement(P, 0.002);
    AMSpleen->AddElement(S, 0.002);
    AMSpleen->AddElement(Cl, 0.003);
    AMSpleen->AddElement(K, 0.002);
    AMSpleen->AddElement(Ca, 0.);
    AMSpleen->AddElement(Fe, 0.001);
    AMSpleen->AddElement(I, 0.);
    AMSpleen->AddElement(Gd, 0.);

    rindexAMSpleen = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMSpleen);
    
    absLengthAMSpleen = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMSpleen.dat", absLengthAMSpleen);
    
    G4MaterialPropertiesTable *mptAMSpleen = DefineNonScintillatingMaterial(AMSpleen, 14, rindexAMSpleen, absLengthAMSpleen);
    
    // Thyroid
    AMThyroid = new G4Material("AMThyroid", 1.04*g/cm3, 14);
    AMThyroid->AddElement(H, 0.104);
    AMThyroid->AddElement(C, 0.117);
    AMThyroid->AddElement(N, 0.026);
    AMThyroid->AddElement(O, 0.745);
    AMThyroid->AddElement(Na, 0.002);
    AMThyroid->AddElement(Mg, 0.);
    AMThyroid->AddElement(P, 0.001);
    AMThyroid->AddElement(S, 0.001);
    AMThyroid->AddElement(Cl, 0.002);
    AMThyroid->AddElement(K, 0.001);
    AMThyroid->AddElement(Ca, 0.);
    AMThyroid->AddElement(Fe, 0.);
    AMThyroid->AddElement(I, 0.001);
    AMThyroid->AddElement(Gd, 0.);

    rindexAMThyroid = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMThyroid);
    
    absLengthAMThyroid = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMThyroid.dat", absLengthAMThyroid);
    
    G4MaterialPropertiesTable *mptAMThyroid = DefineNonScintillatingMaterial(AMThyroid, 14, rindexAMThyroid, absLengthAMThyroid);
    
    // UrinaryBladder
    AMUrinaryBladder = new G4Material("AMUrinaryBladder", 1.04*g/cm3, 14);
    AMUrinaryBladder->AddElement(H, 0.105);
    AMUrinaryBladder->AddElement(C, 0.096);
    AMUrinaryBladder->AddElement(N, 0.026);
    AMUrinaryBladder->AddElement(O, 0.761);
    AMUrinaryBladder->AddElement(Na, 0.002);
    AMUrinaryBladder->AddElement(Mg, 0.);
    AMUrinaryBladder->AddElement(P, 0.002);
    AMUrinaryBladder->AddElement(S, 0.002);
    AMUrinaryBladder->AddElement(Cl, 0.003);
    AMUrinaryBladder->AddElement(K, 0.003);
    AMUrinaryBladder->AddElement(Ca, 0.);
    AMUrinaryBladder->AddElement(Fe, 0.);
    AMUrinaryBladder->AddElement(I, 0.);
    AMUrinaryBladder->AddElement(Gd, 0.);

    rindexAMUrinaryBladder = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMUrinaryBladder);
    
    absLengthAMUrinaryBladder = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMUrinaryBladder.dat", absLengthAMUrinaryBladder);
    
    G4MaterialPropertiesTable *mptAMUrinaryBladder = DefineNonScintillatingMaterial(AMUrinaryBladder, 14, rindexAMUrinaryBladder, absLengthAMUrinaryBladder);
    
    // Testes
    AMTestes = new G4Material("AMTestes", 1.04*g/cm3, 14);
    AMTestes->AddElement(H, 0.106);
    AMTestes->AddElement(C, 0.1);
    AMTestes->AddElement(N, 0.021);
    AMTestes->AddElement(O, 0.764);
    AMTestes->AddElement(Na, 0.002);
    AMTestes->AddElement(Mg, 0.);
    AMTestes->AddElement(P, 0.001);
    AMTestes->AddElement(S, 0.002);
    AMTestes->AddElement(Cl, 0.002);
    AMTestes->AddElement(K, 0.002);
    AMTestes->AddElement(Ca, 0.);
    AMTestes->AddElement(Fe, 0.);
    AMTestes->AddElement(I, 0.);
    AMTestes->AddElement(Gd, 0.);

    rindexAMTestes = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMTestes);
    
    absLengthAMTestes = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMTestes.dat", absLengthAMTestes);
    
    G4MaterialPropertiesTable *mptAMTestes = DefineNonScintillatingMaterial(AMTestes, 14, rindexAMTestes, absLengthAMTestes);
    
    // Adrenals
    AMAdrenals = new G4Material("AMAdrenals", 1.03*g/cm3, 14);
    AMAdrenals->AddElement(H, 0.104);
    AMAdrenals->AddElement(C, 0.221);
    AMAdrenals->AddElement(N, 0.028);
    AMAdrenals->AddElement(O, 0.637);
    AMAdrenals->AddElement(Na, 0.001);
    AMAdrenals->AddElement(Mg, 0.);
    AMAdrenals->AddElement(P, 0.002);
    AMAdrenals->AddElement(S, 0.003);
    AMAdrenals->AddElement(Cl, 0.002);
    AMAdrenals->AddElement(K, 0.002);
    AMAdrenals->AddElement(Ca, 0.);
    AMAdrenals->AddElement(Fe, 0.);
    AMAdrenals->AddElement(I, 0.);
    AMAdrenals->AddElement(Gd, 0.);

    rindexAMAdrenals = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMAdrenals);
    
    absLengthAMAdrenals = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMAdrenals.dat", absLengthAMAdrenals);
    
    G4MaterialPropertiesTable *mptAMAdrenals = DefineNonScintillatingMaterial(AMAdrenals, 14, rindexAMAdrenals, absLengthAMAdrenals);
    
    // Oesophagus
    AMOesophagus = new G4Material("AMOesophagus", 1.03*g/cm3, 14);
    AMOesophagus->AddElement(H, 0.104);
    AMOesophagus->AddElement(C, 0.213);
    AMOesophagus->AddElement(N, 0.029);
    AMOesophagus->AddElement(O, 0.644);
    AMOesophagus->AddElement(Na, 0.001);
    AMOesophagus->AddElement(Mg, 0.);
    AMOesophagus->AddElement(P, 0.002);
    AMOesophagus->AddElement(S, 0.003);
    AMOesophagus->AddElement(Cl, 0.002);
    AMOesophagus->AddElement(K, 0.002);
    AMOesophagus->AddElement(Ca, 0.);
    AMOesophagus->AddElement(Fe, 0.);
    AMOesophagus->AddElement(I, 0.);
    AMOesophagus->AddElement(Gd, 0.);

    rindexAMOesophagus = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMOesophagus);
    
    absLengthAMOesophagus = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMOesophagus.dat", absLengthAMOesophagus);
    
    G4MaterialPropertiesTable *mptAMOesophagus = DefineNonScintillatingMaterial(AMOesophagus, 14, rindexAMOesophagus, absLengthAMOesophagus);
    
    // Gallbladder
    AMGallbladder = new G4Material("AMGallbladder", 1.03*g/cm3, 14);
    AMGallbladder->AddElement(H, 0.104);
    AMGallbladder->AddElement(C, 0.231);
    AMGallbladder->AddElement(N, 0.028);
    AMGallbladder->AddElement(O, 0.627);
    AMGallbladder->AddElement(Na, 0.001);
    AMGallbladder->AddElement(Mg, 0.);
    AMGallbladder->AddElement(P, 0.002);
    AMGallbladder->AddElement(S, 0.003);
    AMGallbladder->AddElement(Cl, 0.002);
    AMGallbladder->AddElement(K, 0.002);
    AMGallbladder->AddElement(Ca, 0.);
    AMGallbladder->AddElement(Fe, 0.);
    AMGallbladder->AddElement(I, 0.);
    AMGallbladder->AddElement(Gd, 0.);

    rindexAMGallbladder = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMGallbladder);
    
    absLengthAMGallbladder = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMGallbladder.dat", absLengthAMGallbladder);
    
    G4MaterialPropertiesTable *mptAMGallbladder = DefineNonScintillatingMaterial(AMGallbladder, 14, rindexAMGallbladder, absLengthAMGallbladder);
    
    // Prostate
    AMProstate = new G4Material("AMProstate", 1.03*g/cm3, 14);
    AMProstate->AddElement(H, 0.104);
    AMProstate->AddElement(C, 0.231);
    AMProstate->AddElement(N, 0.028);
    AMProstate->AddElement(O, 0.627);
    AMProstate->AddElement(Na, 0.001);
    AMProstate->AddElement(Mg, 0.);
    AMProstate->AddElement(P, 0.002);
    AMProstate->AddElement(S, 0.003);
    AMProstate->AddElement(Cl, 0.002);
    AMProstate->AddElement(K, 0.002);
    AMProstate->AddElement(Ca, 0.);
    AMProstate->AddElement(Fe, 0.);
    AMProstate->AddElement(I, 0.);
    AMProstate->AddElement(Gd, 0.);

    rindexAMProstate = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMProstate);
    
    absLengthAMProstate = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMProstate.dat", absLengthAMProstate);
    
    G4MaterialPropertiesTable *mptAMProstate = DefineNonScintillatingMaterial(AMProstate, 14, rindexAMProstate, absLengthAMProstate);
    
    // Lymph
    AMLymph = new G4Material("AMLymph", 1.03*g/cm3, 14);
    AMLymph->AddElement(H, 0.108);
    AMLymph->AddElement(C, 0.042);
    AMLymph->AddElement(N, 0.011);
    AMLymph->AddElement(O, 0.831);
    AMLymph->AddElement(Na, 0.003);
    AMLymph->AddElement(Mg, 0.);
    AMLymph->AddElement(P, 0.);
    AMLymph->AddElement(S, 0.001);
    AMLymph->AddElement(Cl, 0.004);
    AMLymph->AddElement(K, 0.);
    AMLymph->AddElement(Ca, 0.);
    AMLymph->AddElement(Fe, 0.);
    AMLymph->AddElement(I, 0.);
    AMLymph->AddElement(Gd, 0.);

    rindexAMLymph = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMLymph);
    
    absLengthAMLymph = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMLymph.dat", absLengthAMLymph);
    
    G4MaterialPropertiesTable *mptAMLymph = DefineNonScintillatingMaterial(AMLymph, 14, rindexAMLymph, absLengthAMLymph);
    
    // Breast
    AMBreast = new G4Material("AMBreast", 1.02*g/cm3, 14);
    AMBreast->AddElement(H, 0.112);
    AMBreast->AddElement(C, 0.516);
    AMBreast->AddElement(N, 0.011);
    AMBreast->AddElement(O, 0.358);
    AMBreast->AddElement(Na, 0.001);
    AMBreast->AddElement(Mg, 0.);
    AMBreast->AddElement(P, 0.);
    AMBreast->AddElement(S, 0.001);
    AMBreast->AddElement(Cl, 0.001);
    AMBreast->AddElement(K, 0.);
    AMBreast->AddElement(Ca, 0.);
    AMBreast->AddElement(Fe, 0.);
    AMBreast->AddElement(I, 0.);
    AMBreast->AddElement(Gd, 0.);

    rindexAMBreast = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMBreast);
    
    absLengthAMBreast = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMBreast.dat", absLengthAMBreast);
    
    G4MaterialPropertiesTable *mptAMBreast = DefineNonScintillatingMaterial(AMBreast, 14, rindexAMBreast, absLengthAMBreast);
    
    // AdiposeTissue
    AMAdiposeTissue = new G4Material("AMAdiposeTissue", 0.95*g/cm3, 14);
    AMAdiposeTissue->AddElement(H, 0.114);
    AMAdiposeTissue->AddElement(C, 0.588);
    AMAdiposeTissue->AddElement(N, 0.008);
    AMAdiposeTissue->AddElement(O, 0.287);
    AMAdiposeTissue->AddElement(Na, 0.001);
    AMAdiposeTissue->AddElement(Mg, 0.);
    AMAdiposeTissue->AddElement(P, 0.);
    AMAdiposeTissue->AddElement(S, 0.001);
    AMAdiposeTissue->AddElement(Cl, 0.001);
    AMAdiposeTissue->AddElement(K, 0.);
    AMAdiposeTissue->AddElement(Ca, 0.);
    AMAdiposeTissue->AddElement(Fe, 0.);
    AMAdiposeTissue->AddElement(I, 0.);
    AMAdiposeTissue->AddElement(Gd, 0.);

    rindexAMAdiposeTissue = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMAdiposeTissue);
    
    absLengthAMAdiposeTissue = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMAdiposeTissue.dat", absLengthAMAdiposeTissue);
    
    G4MaterialPropertiesTable *mptAMAdiposeTissue = DefineNonScintillatingMaterial(AMAdiposeTissue, 14, rindexAMAdiposeTissue, absLengthAMAdiposeTissue);
    
    // Lung
    AMLung = new G4Material("AMLung", 0.382*g/cm3, 14);
    AMLung->AddElement(H, 0.103);
    AMLung->AddElement(C, 0.107);
    AMLung->AddElement(N, 0.032);
    AMLung->AddElement(O, 0.746);
    AMLung->AddElement(Na, 0.002);
    AMLung->AddElement(Mg, 0.);
    AMLung->AddElement(P, 0.002);
    AMLung->AddElement(S, 0.003);
    AMLung->AddElement(Cl, 0.003);
    AMLung->AddElement(K, 0.002);
    AMLung->AddElement(Ca, 0.);
    AMLung->AddElement(Fe, 0.);
    AMLung->AddElement(I, 0.);
    AMLung->AddElement(Gd, 0.);

    rindexAMLung = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMLung);
    
    absLengthAMLung = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMLung.dat", absLengthAMLung);
    
    G4MaterialPropertiesTable *mptAMLung = DefineNonScintillatingMaterial(AMLung, 14, rindexAMLung, absLengthAMLung);
    
    // GastroIntestinalContents
    AMGastroIntestinalContents = new G4Material("AMGastroIntestinalContents", 1.04*g/cm3, 14);
    AMGastroIntestinalContents->AddElement(H, 0.1);
    AMGastroIntestinalContents->AddElement(C, 0.222);
    AMGastroIntestinalContents->AddElement(N, 0.022);
    AMGastroIntestinalContents->AddElement(O, 0.644);
    AMGastroIntestinalContents->AddElement(Na, 0.001);
    AMGastroIntestinalContents->AddElement(Mg, 0.);
    AMGastroIntestinalContents->AddElement(P, 0.002);
    AMGastroIntestinalContents->AddElement(S, 0.003);
    AMGastroIntestinalContents->AddElement(Cl, 0.001);
    AMGastroIntestinalContents->AddElement(K, 0.004);
    AMGastroIntestinalContents->AddElement(Ca, 0.001);
    AMGastroIntestinalContents->AddElement(Fe, 0.);
    AMGastroIntestinalContents->AddElement(I, 0.);
    AMGastroIntestinalContents->AddElement(Gd, 0.);

    rindexAMGastroIntestinalContents = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMGastroIntestinalContents);
    
    absLengthAMGastroIntestinalContents = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMGastroIntestinalContents.dat", absLengthAMGastroIntestinalContents);
    
    G4MaterialPropertiesTable *mptAMGastroIntestinalContents = DefineNonScintillatingMaterial(AMGastroIntestinalContents, 14, rindexAMGastroIntestinalContents, absLengthAMGastroIntestinalContents);
    
    // Urine
    AMUrine = new G4Material("AMUrine", 1.04*g/cm3, 14);
    AMUrine->AddElement(H, 0.107);
    AMUrine->AddElement(C, 0.003);
    AMUrine->AddElement(N, 0.01);
    AMUrine->AddElement(O, 0.873);
    AMUrine->AddElement(Na, 0.004);
    AMUrine->AddElement(Mg, 0.);
    AMUrine->AddElement(P, 0.001);
    AMUrine->AddElement(S, 0.);
    AMUrine->AddElement(Cl, 0.);
    AMUrine->AddElement(K, 0.002);
    AMUrine->AddElement(Ca, 0.);
    AMUrine->AddElement(Fe, 0.);
    AMUrine->AddElement(I, 0.);
    AMUrine->AddElement(Gd, 0.);

    rindexAMUrine = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMUrine);
    
    absLengthAMUrine = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMUrine.dat", absLengthAMUrine);
    
    G4MaterialPropertiesTable *mptAMUrine = DefineNonScintillatingMaterial(AMUrine, 14, rindexAMUrine, absLengthAMUrine);
    
    // Air
    AMAir = new G4Material("AMAir", 0.001*g/cm3, 14);
    AMAir->AddElement(H, 0.);
    AMAir->AddElement(C, 0.);
    AMAir->AddElement(N, 0.8);
    AMAir->AddElement(O, 0.2);
    AMAir->AddElement(Na, 0.);
    AMAir->AddElement(Mg, 0.);
    AMAir->AddElement(P, 0.);
    AMAir->AddElement(S, 0.);
    AMAir->AddElement(Cl, 0.);
    AMAir->AddElement(K, 0.);
    AMAir->AddElement(Ca, 0.);
    AMAir->AddElement(Fe, 0.);
    AMAir->AddElement(I, 0.);
    AMAir->AddElement(Gd, 0.);

    rindexAMAir = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMAir);
    
    absLengthAMAir = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMAir.dat", absLengthAMAir);
    
    G4MaterialPropertiesTable *mptAMAir = DefineNonScintillatingMaterial(AMAir, 14, rindexAMAir, absLengthAMAir);
    
    // IBlood
    AMIBlood = new G4Material("AMIBlood", 1.06*g/cm3, 14);
    AMIBlood->AddElement(H, 0.102*(1-IConcentration));
    AMIBlood->AddElement(C, 0.11*(1-IConcentration));
    AMIBlood->AddElement(N, 0.033*(1-IConcentration));
    AMIBlood->AddElement(O, 0.745*(1-IConcentration));
    AMIBlood->AddElement(Na, 0.001*(1-IConcentration));
    AMIBlood->AddElement(Mg, 0.);
    AMIBlood->AddElement(P, 0.001*(1-IConcentration));
    AMIBlood->AddElement(S, 0.002*(1-IConcentration));
    AMIBlood->AddElement(Cl, 0.003*(1-IConcentration));
    AMIBlood->AddElement(K, 0.002*(1-IConcentration));
    AMIBlood->AddElement(Ca, 0.);
    AMIBlood->AddElement(Fe, 0.001*(1-IConcentration));
    AMIBlood->AddElement(I, IConcentration);
    AMIBlood->AddElement(Gd, 0.);

    rindexAMIBlood = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMIBlood);
    
    absLengthAMIBlood = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMIBlood.dat", absLengthAMIBlood);
    
    G4MaterialPropertiesTable *mptAMIBlood = DefineNonScintillatingMaterial(AMIBlood, 14, rindexAMIBlood, absLengthAMIBlood);
    
    // GdBlood
    AMGdBlood = new G4Material("AMGdBlood", 1.06*g/cm3, 14);
    AMGdBlood->AddElement(H, 0.102*(1-GdConcentration));
    AMGdBlood->AddElement(C, 0.11*(1-GdConcentration));
    AMGdBlood->AddElement(N, 0.033*(1-GdConcentration));
    AMGdBlood->AddElement(O, 0.745*(1-GdConcentration));
    AMGdBlood->AddElement(Na, 0.001*(1-GdConcentration));
    AMGdBlood->AddElement(Mg, 0.);
    AMGdBlood->AddElement(P, 0.001*(1-GdConcentration));
    AMGdBlood->AddElement(S, 0.002*(1-GdConcentration));
    AMGdBlood->AddElement(Cl, 0.003*(1-GdConcentration));
    AMGdBlood->AddElement(K, 0.002*(1-GdConcentration));
    AMGdBlood->AddElement(Ca, 0.);
    AMGdBlood->AddElement(Fe, 0.001*(1-GdConcentration));
    AMGdBlood->AddElement(I, 0.);
    AMGdBlood->AddElement(Gd, GdConcentration);

    rindexAMGdBlood = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAMGdBlood);
    
    absLengthAMGdBlood = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAMGdBlood.dat", absLengthAMGdBlood);
    
    G4MaterialPropertiesTable *mptAMGdBlood = DefineNonScintillatingMaterial(AMGdBlood, 14, rindexAMGdBlood, absLengthAMGdBlood);
}

void NSDetectorConstruction::DefineAFBioMedia()
{
    // Teeth
    AFTeeth = new G4Material("AFTeeth", 2.75*g/cm3, 13);
    AFTeeth->AddElement(H, 0.022);
    AFTeeth->AddElement(C, 0.095);
    AFTeeth->AddElement(N, 0.029);
    AFTeeth->AddElement(O, 0.421);
    AFTeeth->AddElement(Na, 0.);
    AFTeeth->AddElement(Mg, 0.007);
    AFTeeth->AddElement(P, 0.137);
    AFTeeth->AddElement(S, 0.);
    AFTeeth->AddElement(Cl, 0.);
    AFTeeth->AddElement(K, 0.);
    AFTeeth->AddElement(Ca, 0.289);
    AFTeeth->AddElement(Fe, 0.);
    AFTeeth->AddElement(I, 0.);

    rindexAFTeeth = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFTeeth);
    
    absLengthAFTeeth = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFTeeth.dat", absLengthAFTeeth);
    
    G4MaterialPropertiesTable *mptAFTeeth = DefineNonScintillatingMaterial(AFTeeth, 13, rindexAFTeeth, absLengthAFTeeth);
    
    // MineralBone
    AFMineralBone = new G4Material("AFMineralBone", 1.92*g/cm3, 13);
    AFMineralBone->AddElement(H, 0.036);
    AFMineralBone->AddElement(C, 0.159);
    AFMineralBone->AddElement(N, 0.042);
    AFMineralBone->AddElement(O, 0.448);
    AFMineralBone->AddElement(Na, 0.003);
    AFMineralBone->AddElement(Mg, 0.002);
    AFMineralBone->AddElement(P, 0.094);
    AFMineralBone->AddElement(S, 0.003);
    AFMineralBone->AddElement(Cl, 0.);
    AFMineralBone->AddElement(K, 0.);
    AFMineralBone->AddElement(Ca, 0.213);
    AFMineralBone->AddElement(Fe, 0.);
    AFMineralBone->AddElement(I, 0.);

    rindexAFMineralBone = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFMineralBone);
    
    absLengthAFMineralBone = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFMineralBone.dat", absLengthAFMineralBone);
    
    G4MaterialPropertiesTable *mptAFMineralBone = DefineNonScintillatingMaterial(AFMineralBone, 13, rindexAFMineralBone, absLengthAFMineralBone);
    
    // HumeriUpper
    AFHumeriUpper = new G4Material("AFHumeriUpper", 1.185*g/cm3, 13);
    AFHumeriUpper->AddElement(H, 0.087);
    AFHumeriUpper->AddElement(C, 0.366);
    AFHumeriUpper->AddElement(N, 0.025);
    AFHumeriUpper->AddElement(O, 0.422);
    AFHumeriUpper->AddElement(Na, 0.002);
    AFHumeriUpper->AddElement(Mg, 0.001);
    AFHumeriUpper->AddElement(P, 0.03);
    AFHumeriUpper->AddElement(S, 0.003);
    AFHumeriUpper->AddElement(Cl, 0.001);
    AFHumeriUpper->AddElement(K, 0.001);
    AFHumeriUpper->AddElement(Ca, 0.062);
    AFHumeriUpper->AddElement(Fe, 0.);
    AFHumeriUpper->AddElement(I, 0.);

    rindexAFHumeriUpper = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFHumeriUpper);
    
    absLengthAFHumeriUpper = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFHumeriUpper.dat", absLengthAFHumeriUpper);
    
    G4MaterialPropertiesTable *mptAFHumeriUpper = DefineNonScintillatingMaterial(AFHumeriUpper, 13, rindexAFHumeriUpper, absLengthAFHumeriUpper);
    
    // HumeriLower
    AFHumeriLower = new G4Material("AFHumeriLower", 1.117*g/cm3, 13);
    AFHumeriLower->AddElement(H, 0.096);
    AFHumeriLower->AddElement(C, 0.473);
    AFHumeriLower->AddElement(N, 0.017);
    AFHumeriLower->AddElement(O, 0.341);
    AFHumeriLower->AddElement(Na, 0.002);
    AFHumeriLower->AddElement(Mg, 0.);
    AFHumeriLower->AddElement(P, 0.022);
    AFHumeriLower->AddElement(S, 0.002);
    AFHumeriLower->AddElement(Cl, 0.001);
    AFHumeriLower->AddElement(K, 0.);
    AFHumeriLower->AddElement(Ca, 0.046);
    AFHumeriLower->AddElement(Fe, 0.);
    AFHumeriLower->AddElement(I, 0.);

    rindexAFHumeriLower = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFHumeriLower);
    
    absLengthAFHumeriLower = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFHumeriLower.dat", absLengthAFHumeriLower);
    
    G4MaterialPropertiesTable *mptAFHumeriLower = DefineNonScintillatingMaterial(AFHumeriLower, 13, rindexAFHumeriLower, absLengthAFHumeriLower);
    
    // LowerArmBone
    AFLowerArmBone = new G4Material("AFLowerArmBone", 1.117*g/cm3, 13);
    AFLowerArmBone->AddElement(H, 0.096);
    AFLowerArmBone->AddElement(C, 0.473);
    AFLowerArmBone->AddElement(N, 0.017);
    AFLowerArmBone->AddElement(O, 0.341);
    AFLowerArmBone->AddElement(Na, 0.002);
    AFLowerArmBone->AddElement(Mg, 0.);
    AFLowerArmBone->AddElement(P, 0.022);
    AFLowerArmBone->AddElement(S, 0.002);
    AFLowerArmBone->AddElement(Cl, 0.001);
    AFLowerArmBone->AddElement(K, 0.);
    AFLowerArmBone->AddElement(Ca, 0.046);
    AFLowerArmBone->AddElement(Fe, 0.);
    AFLowerArmBone->AddElement(I, 0.);

    rindexAFLowerArmBone = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFLowerArmBone);
    
    absLengthAFLowerArmBone = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFLowerArmBone.dat", absLengthAFLowerArmBone);
    
    G4MaterialPropertiesTable *mptAFLowerArmBone = DefineNonScintillatingMaterial(AFLowerArmBone, 13, rindexAFLowerArmBone, absLengthAFLowerArmBone);
    
    // HandBone
    AFHandBone = new G4Material("AFHandBone", 1.117*g/cm3, 13);
    AFHandBone->AddElement(H, 0.096);
    AFHandBone->AddElement(C, 0.473);
    AFHandBone->AddElement(N, 0.017);
    AFHandBone->AddElement(O, 0.341);
    AFHandBone->AddElement(Na, 0.002);
    AFHandBone->AddElement(Mg, 0.);
    AFHandBone->AddElement(P, 0.022);
    AFHandBone->AddElement(S, 0.002);
    AFHandBone->AddElement(Cl, 0.001);
    AFHandBone->AddElement(K, 0.);
    AFHandBone->AddElement(Ca, 0.046);
    AFHandBone->AddElement(Fe, 0.);
    AFHandBone->AddElement(I, 0.);

    rindexAFHandBone = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFHandBone);
    
    absLengthAFHandBone = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFHandBone.dat", absLengthAFHandBone);
    
    G4MaterialPropertiesTable *mptAFHandBone = DefineNonScintillatingMaterial(AFHandBone, 13, rindexAFHandBone, absLengthAFHandBone);
    
    // Clavicles
    AFClavicles = new G4Material("AFClavicles", 1.191*g/cm3, 13);
    AFClavicles->AddElement(H, 0.087);
    AFClavicles->AddElement(C, 0.361);
    AFClavicles->AddElement(N, 0.025);
    AFClavicles->AddElement(O, 0.424);
    AFClavicles->AddElement(Na, 0.002);
    AFClavicles->AddElement(Mg, 0.001);
    AFClavicles->AddElement(P, 0.031);
    AFClavicles->AddElement(S, 0.003);
    AFClavicles->AddElement(Cl, 0.001);
    AFClavicles->AddElement(K, 0.001);
    AFClavicles->AddElement(Ca, 0.064);
    AFClavicles->AddElement(Fe, 0.);
    AFClavicles->AddElement(I, 0.);

    rindexAFClavicles = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFClavicles);
    
    absLengthAFClavicles = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFClavicles.dat", absLengthAFClavicles);
    
    G4MaterialPropertiesTable *mptAFClavicles = DefineNonScintillatingMaterial(AFClavicles, 13, rindexAFClavicles, absLengthAFClavicles);
    
    // Cranium
    AFCranium = new G4Material("AFCranium", 1.245*g/cm3, 13);
    AFCranium->AddElement(H, 0.081);
    AFCranium->AddElement(C, 0.317);
    AFCranium->AddElement(N, 0.028);
    AFCranium->AddElement(O, 0.451);
    AFCranium->AddElement(Na, 0.002);
    AFCranium->AddElement(Mg, 0.001);
    AFCranium->AddElement(P, 0.037);
    AFCranium->AddElement(S, 0.003);
    AFCranium->AddElement(Cl, 0.001);
    AFCranium->AddElement(K, 0.001);
    AFCranium->AddElement(Ca, 0.078);
    AFCranium->AddElement(Fe, 0.);
    AFCranium->AddElement(I, 0.);

    rindexAFCranium = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFCranium);
    
    absLengthAFCranium = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFCranium.dat", absLengthAFCranium);
    
    G4MaterialPropertiesTable *mptAFCranium = DefineNonScintillatingMaterial(AFCranium, 13, rindexAFCranium, absLengthAFCranium);
    
    // FemoraUpper
    AFFemoraUpper = new G4Material("AFFemoraUpper", 1.046*g/cm3, 13);
    AFFemoraUpper->AddElement(H, 0.104);
    AFFemoraUpper->AddElement(C, 0.496);
    AFFemoraUpper->AddElement(N, 0.018);
    AFFemoraUpper->AddElement(O, 0.349);
    AFFemoraUpper->AddElement(Na, 0.001);
    AFFemoraUpper->AddElement(Mg, 0.);
    AFFemoraUpper->AddElement(P, 0.009);
    AFFemoraUpper->AddElement(S, 0.002);
    AFFemoraUpper->AddElement(Cl, 0.001);
    AFFemoraUpper->AddElement(K, 0.001);
    AFFemoraUpper->AddElement(Ca, 0.019);
    AFFemoraUpper->AddElement(Fe, 0.);
    AFFemoraUpper->AddElement(I, 0.);

    rindexAFFemoraUpper = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFFemoraUpper);
    
    absLengthAFFemoraUpper = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFFemoraUpper.dat", absLengthAFFemoraUpper);
    
    G4MaterialPropertiesTable *mptAFFemoraUpper = DefineNonScintillatingMaterial(AFFemoraUpper, 13, rindexAFFemoraUpper, absLengthAFFemoraUpper);
    
    // FemoraLower
    AFFemoraLower = new G4Material("AFFemoraLower", 1.117*g/cm3, 13);
    AFFemoraLower->AddElement(H, 0.096);
    AFFemoraLower->AddElement(C, 0.473);
    AFFemoraLower->AddElement(N, 0.017);
    AFFemoraLower->AddElement(O, 0.341);
    AFFemoraLower->AddElement(Na, 0.002);
    AFFemoraLower->AddElement(Mg, 0.);
    AFFemoraLower->AddElement(P, 0.022);
    AFFemoraLower->AddElement(S, 0.002);
    AFFemoraLower->AddElement(Cl, 0.001);
    AFFemoraLower->AddElement(K, 0.);
    AFFemoraLower->AddElement(Ca, 0.046);
    AFFemoraLower->AddElement(Fe, 0.);
    AFFemoraLower->AddElement(I, 0.);

    rindexAFFemoraLower = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFFemoraLower);
    
    absLengthAFFemoraLower = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFFemoraLower.dat", absLengthAFFemoraLower);
    
    G4MaterialPropertiesTable *mptAFFemoraLower = DefineNonScintillatingMaterial(AFFemoraLower, 13, rindexAFFemoraLower, absLengthAFFemoraLower);
    
    // LowerLeg
    AFLowerLeg = new G4Material("AFLowerLeg", 1.117*g/cm3, 13);
    AFLowerLeg->AddElement(H, 0.096);
    AFLowerLeg->AddElement(C, 0.473);
    AFLowerLeg->AddElement(N, 0.017);
    AFLowerLeg->AddElement(O, 0.341);
    AFLowerLeg->AddElement(Na, 0.002);
    AFLowerLeg->AddElement(Mg, 0.);
    AFLowerLeg->AddElement(P, 0.022);
    AFLowerLeg->AddElement(S, 0.002);
    AFLowerLeg->AddElement(Cl, 0.001);
    AFLowerLeg->AddElement(K, 0.);
    AFLowerLeg->AddElement(Ca, 0.046);
    AFLowerLeg->AddElement(Fe, 0.);
    AFLowerLeg->AddElement(I, 0.);

    rindexAFLowerLeg = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFLowerLeg);
    
    absLengthAFLowerLeg = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFLowerLeg.dat", absLengthAFLowerLeg);
    
    G4MaterialPropertiesTable *mptAFLowerLeg = DefineNonScintillatingMaterial(AFLowerLeg, 13, rindexAFLowerLeg, absLengthAFLowerLeg);
    
    // Foot
    AFFoot = new G4Material("AFFoot", 1.117*g/cm3, 13);
    AFFoot->AddElement(H, 0.096);
    AFFoot->AddElement(C, 0.473);
    AFFoot->AddElement(N, 0.017);
    AFFoot->AddElement(O, 0.341);
    AFFoot->AddElement(Na, 0.002);
    AFFoot->AddElement(Mg, 0.);
    AFFoot->AddElement(P, 0.022);
    AFFoot->AddElement(S, 0.002);
    AFFoot->AddElement(Cl, 0.001);
    AFFoot->AddElement(K, 0.);
    AFFoot->AddElement(Ca, 0.046);
    AFFoot->AddElement(Fe, 0.);
    AFFoot->AddElement(I, 0.);

    rindexAFFoot = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFFoot);
    
    absLengthAFFoot = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFFoot.dat", absLengthAFFoot);
    
    G4MaterialPropertiesTable *mptAFFoot = DefineNonScintillatingMaterial(AFFoot, 13, rindexAFFoot, absLengthAFFoot);
    
    // Mandible
    AFMandible = new G4Material("AFMandible", 1.189*g/cm3, 13);
    AFMandible->AddElement(H, 0.087);
    AFMandible->AddElement(C, 0.357);
    AFMandible->AddElement(N, 0.026);
    AFMandible->AddElement(O, 0.429);
    AFMandible->AddElement(Na, 0.002);
    AFMandible->AddElement(Mg, 0.001);
    AFMandible->AddElement(P, 0.03);
    AFMandible->AddElement(S, 0.003);
    AFMandible->AddElement(Cl, 0.001);
    AFMandible->AddElement(K, 0.001);
    AFMandible->AddElement(Ca, 0.063);
    AFMandible->AddElement(Fe, 0.);
    AFMandible->AddElement(I, 0.);

    rindexAFMandible = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFMandible);
    
    absLengthAFMandible = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFMandible.dat", absLengthAFMandible);
    
    G4MaterialPropertiesTable *mptAFMandible = DefineNonScintillatingMaterial(AFMandible, 13, rindexAFMandible, absLengthAFMandible);
    
    // Pelvis
    AFPelvis = new G4Material("AFPelvis", 1.109*g/cm3, 13);
    AFPelvis->AddElement(H, 0.096);
    AFPelvis->AddElement(C, 0.406);
    AFPelvis->AddElement(N, 0.025);
    AFPelvis->AddElement(O, 0.412);
    AFPelvis->AddElement(Na, 0.001);
    AFPelvis->AddElement(Mg, 0.);
    AFPelvis->AddElement(P, 0.018);
    AFPelvis->AddElement(S, 0.002);
    AFPelvis->AddElement(Cl, 0.001);
    AFPelvis->AddElement(K, 0.001);
    AFPelvis->AddElement(Ca, 0.038);
    AFPelvis->AddElement(Fe, 0.);
    AFPelvis->AddElement(I, 0.);

    rindexAFPelvis = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFPelvis);
    
    absLengthAFPelvis = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFPelvis.dat", absLengthAFPelvis);
    
    G4MaterialPropertiesTable *mptAFPelvis = DefineNonScintillatingMaterial(AFPelvis, 13, rindexAFPelvis, absLengthAFPelvis);
    
    // Ribs
    AFRibs = new G4Material("AFRibs", 1.092*g/cm3, 13);
    AFRibs->AddElement(H, 0.097);
    AFRibs->AddElement(C, 0.381);
    AFRibs->AddElement(N, 0.028);
    AFRibs->AddElement(O, 0.445);
    AFRibs->AddElement(Na, 0.001);
    AFRibs->AddElement(Mg, 0.);
    AFRibs->AddElement(P, 0.014);
    AFRibs->AddElement(S, 0.002);
    AFRibs->AddElement(Cl, 0.002);
    AFRibs->AddElement(K, 0.001);
    AFRibs->AddElement(Ca, 0.028);
    AFRibs->AddElement(Fe, 0.001);
    AFRibs->AddElement(I, 0.);

    rindexAFRibs = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFRibs);
    
    absLengthAFRibs = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFRibs.dat", absLengthAFRibs);
    
    G4MaterialPropertiesTable *mptAFRibs = DefineNonScintillatingMaterial(AFRibs, 13, rindexAFRibs, absLengthAFRibs);
    
    // Scapulae
    AFScapulae = new G4Material("AFScapulae", 1.128*g/cm3, 13);
    AFScapulae->AddElement(H, 0.094);
    AFScapulae->AddElement(C, 0.406);
    AFScapulae->AddElement(N, 0.024);
    AFScapulae->AddElement(O, 0.404);
    AFScapulae->AddElement(Na, 0.001);
    AFScapulae->AddElement(Mg, 0.);
    AFScapulae->AddElement(P, 0.022);
    AFScapulae->AddElement(S, 0.002);
    AFScapulae->AddElement(Cl, 0.001);
    AFScapulae->AddElement(K, 0.001);
    AFScapulae->AddElement(Ca, 0.045);
    AFScapulae->AddElement(Fe, 0.);
    AFScapulae->AddElement(I, 0.);

    rindexAFScapulae = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFScapulae);
    
    absLengthAFScapulae = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFScapulae.dat", absLengthAFScapulae);
    
    G4MaterialPropertiesTable *mptAFScapulae = DefineNonScintillatingMaterial(AFScapulae, 13, rindexAFScapulae, absLengthAFScapulae);
    
    // CervicalSpine
    AFCervicalSpine = new G4Material("AFCervicalSpine", 1.135*g/cm3, 13);
    AFCervicalSpine->AddElement(H, 0.092);
    AFCervicalSpine->AddElement(C, 0.351);
    AFCervicalSpine->AddElement(N, 0.029);
    AFCervicalSpine->AddElement(O, 0.458);
    AFCervicalSpine->AddElement(Na, 0.001);
    AFCervicalSpine->AddElement(Mg, 0.);
    AFCervicalSpine->AddElement(P, 0.021);
    AFCervicalSpine->AddElement(S, 0.002);
    AFCervicalSpine->AddElement(Cl, 0.002);
    AFCervicalSpine->AddElement(K, 0.001);
    AFCervicalSpine->AddElement(Ca, 0.043);
    AFCervicalSpine->AddElement(Fe, 0.);
    AFCervicalSpine->AddElement(I, 0.);

    rindexAFCervicalSpine = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFCervicalSpine);
    
    absLengthAFCervicalSpine = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFCervicalSpine.dat", absLengthAFCervicalSpine);
    
    G4MaterialPropertiesTable *mptAFCervicalSpine = DefineNonScintillatingMaterial(AFCervicalSpine, 13, rindexAFCervicalSpine, absLengthAFCervicalSpine);
    
    // ThoracicSpine
    AFThoracicSpine = new G4Material("AFThoracicSpine", 1.084*g/cm3, 13);
    AFThoracicSpine->AddElement(H, 0.098);
    AFThoracicSpine->AddElement(C, 0.386);
    AFThoracicSpine->AddElement(N, 0.028);
    AFThoracicSpine->AddElement(O, 0.442);
    AFThoracicSpine->AddElement(Na, 0.001);
    AFThoracicSpine->AddElement(Mg, 0.);
    AFThoracicSpine->AddElement(P, 0.013);
    AFThoracicSpine->AddElement(S, 0.002);
    AFThoracicSpine->AddElement(Cl, 0.002);
    AFThoracicSpine->AddElement(K, 0.001);
    AFThoracicSpine->AddElement(Ca, 0.026);
    AFThoracicSpine->AddElement(Fe, 0.001);
    AFThoracicSpine->AddElement(I, 0.);

    rindexAFThoracicSpine = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFThoracicSpine);
    
    absLengthAFThoracicSpine = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFThoracicSpine.dat", absLengthAFThoracicSpine);
    
    G4MaterialPropertiesTable *mptAFThoracicSpine = DefineNonScintillatingMaterial(AFThoracicSpine, 13, rindexAFThoracicSpine, absLengthAFThoracicSpine);
    
    // LumbarSpine
    AFLumbarSpine = new G4Material("AFLumbarSpine", 1.171*g/cm3, 13);
    AFLumbarSpine->AddElement(H, 0.088);
    AFLumbarSpine->AddElement(C, 0.329);
    AFLumbarSpine->AddElement(N, 0.03);
    AFLumbarSpine->AddElement(O, 0.466);
    AFLumbarSpine->AddElement(Na, 0.001);
    AFLumbarSpine->AddElement(Mg, 0.001);
    AFLumbarSpine->AddElement(P, 0.026);
    AFLumbarSpine->AddElement(S, 0.003);
    AFLumbarSpine->AddElement(Cl, 0.001);
    AFLumbarSpine->AddElement(K, 0.001);
    AFLumbarSpine->AddElement(Ca, 0.054);
    AFLumbarSpine->AddElement(Fe, 0.);
    AFLumbarSpine->AddElement(I, 0.);

    rindexAFLumbarSpine = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFLumbarSpine);
    
    absLengthAFLumbarSpine = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFLumbarSpine.dat", absLengthAFLumbarSpine);
    
    G4MaterialPropertiesTable *mptAFLumbarSpine = DefineNonScintillatingMaterial(AFLumbarSpine, 13, rindexAFLumbarSpine, absLengthAFLumbarSpine);
    
    // Sacrum
    AFSacrum = new G4Material("AFSacrum", 1.052*g/cm3, 13);
    AFSacrum->AddElement(H, 0.102);
    AFSacrum->AddElement(C, 0.41);
    AFSacrum->AddElement(N, 0.027);
    AFSacrum->AddElement(O, 0.433);
    AFSacrum->AddElement(Na, 0.001);
    AFSacrum->AddElement(Mg, 0.);
    AFSacrum->AddElement(P, 0.007);
    AFSacrum->AddElement(S, 0.002);
    AFSacrum->AddElement(Cl, 0.002);
    AFSacrum->AddElement(K, 0.001);
    AFSacrum->AddElement(Ca, 0.014);
    AFSacrum->AddElement(Fe, 0.001);
    AFSacrum->AddElement(I, 0.);

    rindexAFSacrum = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFSacrum);
    
    absLengthAFSacrum = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFSacrum.dat", absLengthAFSacrum);
    
    G4MaterialPropertiesTable *mptAFSacrum = DefineNonScintillatingMaterial(AFSacrum, 13, rindexAFSacrum, absLengthAFSacrum);
    
    // Sternum
    AFSternum = new G4Material("AFSternum", 1.076*g/cm3, 13);
    AFSternum->AddElement(H, 0.099);
    AFSternum->AddElement(C, 0.392);
    AFSternum->AddElement(N, 0.028);
    AFSternum->AddElement(O, 0.439);
    AFSternum->AddElement(Na, 0.001);
    AFSternum->AddElement(Mg, 0.);
    AFSternum->AddElement(P, 0.012);
    AFSternum->AddElement(S, 0.002);
    AFSternum->AddElement(Cl, 0.002);
    AFSternum->AddElement(K, 0.001);
    AFSternum->AddElement(Ca, 0.023);
    AFSternum->AddElement(Fe, 0.001);
    AFSternum->AddElement(I, 0.);

    rindexAFSternum = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFSternum);
    
    absLengthAFSternum = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFSternum.dat", absLengthAFSternum);
    
    G4MaterialPropertiesTable *mptAFSternum = DefineNonScintillatingMaterial(AFSternum, 13, rindexAFSternum, absLengthAFSternum);
    
    // HumeriFemoraUpperMedullaryCavity
    AFHumeriFemoraUpperMedullaryCavity = new G4Material("AFHumeriFemoraUpperMedullaryCavity", 0.98*g/cm3, 13);
    AFHumeriFemoraUpperMedullaryCavity->AddElement(H, 0.115);
    AFHumeriFemoraUpperMedullaryCavity->AddElement(C, 0.637);
    AFHumeriFemoraUpperMedullaryCavity->AddElement(N, 0.007);
    AFHumeriFemoraUpperMedullaryCavity->AddElement(O, 0.238);
    AFHumeriFemoraUpperMedullaryCavity->AddElement(Na, 0.001);
    AFHumeriFemoraUpperMedullaryCavity->AddElement(Mg, 0.);
    AFHumeriFemoraUpperMedullaryCavity->AddElement(P, 0.);
    AFHumeriFemoraUpperMedullaryCavity->AddElement(S, 0.001);
    AFHumeriFemoraUpperMedullaryCavity->AddElement(Cl, 0.001);
    AFHumeriFemoraUpperMedullaryCavity->AddElement(K, 0.);
    AFHumeriFemoraUpperMedullaryCavity->AddElement(Ca, 0.);
    AFHumeriFemoraUpperMedullaryCavity->AddElement(Fe, 0.);
    AFHumeriFemoraUpperMedullaryCavity->AddElement(I, 0.);

    rindexAFHumeriFemoraUpperMedullaryCavity = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFHumeriFemoraUpperMedullaryCavity);
    
    absLengthAFHumeriFemoraUpperMedullaryCavity = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFHumeriFemoraUpperMedullaryCavity.dat", absLengthAFHumeriFemoraUpperMedullaryCavity);
    
    G4MaterialPropertiesTable *mptAFHumeriFemoraUpperMedullaryCavity = DefineNonScintillatingMaterial(AFHumeriFemoraUpperMedullaryCavity, 13, rindexAFHumeriFemoraUpperMedullaryCavity, absLengthAFHumeriFemoraUpperMedullaryCavity);
    
    // HumeriFemoraLowerMedullaryCavity
    AFHumeriFemoraLowerMedullaryCavity = new G4Material("AFHumeriFemoraLowerMedullaryCavity", 0.98*g/cm3, 13);
    AFHumeriFemoraLowerMedullaryCavity->AddElement(H, 0.115);
    AFHumeriFemoraLowerMedullaryCavity->AddElement(C, 0.637);
    AFHumeriFemoraLowerMedullaryCavity->AddElement(N, 0.007);
    AFHumeriFemoraLowerMedullaryCavity->AddElement(O, 0.238);
    AFHumeriFemoraLowerMedullaryCavity->AddElement(Na, 0.001);
    AFHumeriFemoraLowerMedullaryCavity->AddElement(Mg, 0.);
    AFHumeriFemoraLowerMedullaryCavity->AddElement(P, 0.);
    AFHumeriFemoraLowerMedullaryCavity->AddElement(S, 0.001);
    AFHumeriFemoraLowerMedullaryCavity->AddElement(Cl, 0.001);
    AFHumeriFemoraLowerMedullaryCavity->AddElement(K, 0.);
    AFHumeriFemoraLowerMedullaryCavity->AddElement(Ca, 0.);
    AFHumeriFemoraLowerMedullaryCavity->AddElement(Fe, 0.);
    AFHumeriFemoraLowerMedullaryCavity->AddElement(I, 0.);

    rindexAFHumeriFemoraLowerMedullaryCavity = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFHumeriFemoraLowerMedullaryCavity);
    
    absLengthAFHumeriFemoraLowerMedullaryCavity = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFHumeriFemoraLowerMedullaryCavity.dat", absLengthAFHumeriFemoraLowerMedullaryCavity);
    
    G4MaterialPropertiesTable *mptAFHumeriFemoraLowerMedullaryCavity = DefineNonScintillatingMaterial(AFHumeriFemoraLowerMedullaryCavity, 13, rindexAFHumeriFemoraLowerMedullaryCavity, absLengthAFHumeriFemoraLowerMedullaryCavity);
    
    // LowerArmMedullaryCavity
    AFLowerArmMedullaryCavity = new G4Material("AFLowerArmMedullaryCavity", 0.98*g/cm3, 13);
    AFLowerArmMedullaryCavity->AddElement(H, 0.115);
    AFLowerArmMedullaryCavity->AddElement(C, 0.637);
    AFLowerArmMedullaryCavity->AddElement(N, 0.007);
    AFLowerArmMedullaryCavity->AddElement(O, 0.238);
    AFLowerArmMedullaryCavity->AddElement(Na, 0.001);
    AFLowerArmMedullaryCavity->AddElement(Mg, 0.);
    AFLowerArmMedullaryCavity->AddElement(P, 0.);
    AFLowerArmMedullaryCavity->AddElement(S, 0.001);
    AFLowerArmMedullaryCavity->AddElement(Cl, 0.001);
    AFLowerArmMedullaryCavity->AddElement(K, 0.);
    AFLowerArmMedullaryCavity->AddElement(Ca, 0.);
    AFLowerArmMedullaryCavity->AddElement(Fe, 0.);
    AFLowerArmMedullaryCavity->AddElement(I, 0.);

    rindexAFLowerArmMedullaryCavity = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFLowerArmMedullaryCavity);
    
    absLengthAFLowerArmMedullaryCavity = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFLowerArmMedullaryCavity.dat", absLengthAFLowerArmMedullaryCavity);
    
    G4MaterialPropertiesTable *mptAFLowerArmMedullaryCavity = DefineNonScintillatingMaterial(AFLowerArmMedullaryCavity, 13, rindexAFLowerArmMedullaryCavity, absLengthAFLowerArmMedullaryCavity);
    
    // LowerLegMedullaryCavity
    AFLowerLegMedullaryCavity = new G4Material("AFLowerLegMedullaryCavity", 0.98*g/cm3, 13);
    AFLowerLegMedullaryCavity->AddElement(H, 0.115);
    AFLowerLegMedullaryCavity->AddElement(C, 0.637);
    AFLowerLegMedullaryCavity->AddElement(N, 0.007);
    AFLowerLegMedullaryCavity->AddElement(O, 0.238);
    AFLowerLegMedullaryCavity->AddElement(Na, 0.001);
    AFLowerLegMedullaryCavity->AddElement(Mg, 0.);
    AFLowerLegMedullaryCavity->AddElement(P, 0.);
    AFLowerLegMedullaryCavity->AddElement(S, 0.001);
    AFLowerLegMedullaryCavity->AddElement(Cl, 0.001);
    AFLowerLegMedullaryCavity->AddElement(K, 0.);
    AFLowerLegMedullaryCavity->AddElement(Ca, 0.);
    AFLowerLegMedullaryCavity->AddElement(Fe, 0.);
    AFLowerLegMedullaryCavity->AddElement(I, 0.);

    rindexAFLowerLegMedullaryCavity = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFLowerLegMedullaryCavity);
    
    absLengthAFLowerLegMedullaryCavity = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFLowerLegMedullaryCavity.dat", absLengthAFLowerLegMedullaryCavity);
    
    G4MaterialPropertiesTable *mptAFLowerLegMedullaryCavity = DefineNonScintillatingMaterial(AFLowerLegMedullaryCavity, 13, rindexAFLowerLegMedullaryCavity, absLengthAFLowerLegMedullaryCavity);
    
    // Cartilage
    AFCartilage = new G4Material("AFCartilage", 1.1*g/cm3, 13);
    AFCartilage->AddElement(H, 0.096);
    AFCartilage->AddElement(C, 0.099);
    AFCartilage->AddElement(N, 0.022);
    AFCartilage->AddElement(O, 0.744);
    AFCartilage->AddElement(Na, 0.005);
    AFCartilage->AddElement(Mg, 0.);
    AFCartilage->AddElement(P, 0.022);
    AFCartilage->AddElement(S, 0.009);
    AFCartilage->AddElement(Cl, 0.003);
    AFCartilage->AddElement(K, 0.);
    AFCartilage->AddElement(Ca, 0.);
    AFCartilage->AddElement(Fe, 0.);
    AFCartilage->AddElement(I, 0.);

    rindexAFCartilage = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFCartilage);
    
    absLengthAFCartilage = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFCartilage.dat", absLengthAFCartilage);
    
    G4MaterialPropertiesTable *mptAFCartilage = DefineNonScintillatingMaterial(AFCartilage, 13, rindexAFCartilage, absLengthAFCartilage);
    
    // Skin
    AFSkin = new G4Material("AFSkin", 1.09*g/cm3, 13);
    AFSkin->AddElement(H, 0.1);
    AFSkin->AddElement(C, 0.199);
    AFSkin->AddElement(N, 0.042);
    AFSkin->AddElement(O, 0.65);
    AFSkin->AddElement(Na, 0.002);
    AFSkin->AddElement(Mg, 0.);
    AFSkin->AddElement(P, 0.001);
    AFSkin->AddElement(S, 0.002);
    AFSkin->AddElement(Cl, 0.003);
    AFSkin->AddElement(K, 0.001);
    AFSkin->AddElement(Ca, 0.);
    AFSkin->AddElement(Fe, 0.);
    AFSkin->AddElement(I, 0.);

    rindexAFSkin = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFSkin);
    
    absLengthAFSkin = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFSkin.dat", absLengthAFSkin);
    
    G4MaterialPropertiesTable *mptAFSkin = DefineNonScintillatingMaterial(AFSkin, 13, rindexAFSkin, absLengthAFSkin);
    
    // Blood
    AFBlood = new G4Material("AFBlood", 1.06*g/cm3, 13);
    AFBlood->AddElement(H, 0.102);
    AFBlood->AddElement(C, 0.11);
    AFBlood->AddElement(N, 0.033);
    AFBlood->AddElement(O, 0.745);
    AFBlood->AddElement(Na, 0.001);
    AFBlood->AddElement(Mg, 0.);
    AFBlood->AddElement(P, 0.001);
    AFBlood->AddElement(S, 0.002);
    AFBlood->AddElement(Cl, 0.003);
    AFBlood->AddElement(K, 0.002);
    AFBlood->AddElement(Ca, 0.);
    AFBlood->AddElement(Fe, 0.001);
    AFBlood->AddElement(I, 0.);

    rindexAFBlood = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFBlood);
    
    absLengthAFBlood = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFBlood.dat", absLengthAFBlood);
    
    G4MaterialPropertiesTable *mptAFBlood = DefineNonScintillatingMaterial(AFBlood, 13, rindexAFBlood, absLengthAFBlood);
    
    // Muscle
    AFMuscle = new G4Material("AFMuscle", 1.05*g/cm3, 13);
    AFMuscle->AddElement(H, 0.102);
    AFMuscle->AddElement(C, 0.142);
    AFMuscle->AddElement(N, 0.034);
    AFMuscle->AddElement(O, 0.711);
    AFMuscle->AddElement(Na, 0.001);
    AFMuscle->AddElement(Mg, 0.);
    AFMuscle->AddElement(P, 0.002);
    AFMuscle->AddElement(S, 0.003);
    AFMuscle->AddElement(Cl, 0.001);
    AFMuscle->AddElement(K, 0.004);
    AFMuscle->AddElement(Ca, 0.);
    AFMuscle->AddElement(Fe, 0.);
    AFMuscle->AddElement(I, 0.);

    rindexAFMuscle = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFMuscle);
    
    absLengthAFMuscle = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFMuscle.dat", absLengthAFMuscle);
    
    G4MaterialPropertiesTable *mptAFMuscle = DefineNonScintillatingMaterial(AFMuscle, 13, rindexAFMuscle, absLengthAFMuscle);
    
    // Liver
    AFLiver = new G4Material("AFLiver", 1.05*g/cm3, 13);
    AFLiver->AddElement(H, 0.102);
    AFLiver->AddElement(C, 0.131);
    AFLiver->AddElement(N, 0.031);
    AFLiver->AddElement(O, 0.724);
    AFLiver->AddElement(Na, 0.002);
    AFLiver->AddElement(Mg, 0.);
    AFLiver->AddElement(P, 0.002);
    AFLiver->AddElement(S, 0.003);
    AFLiver->AddElement(Cl, 0.002);
    AFLiver->AddElement(K, 0.003);
    AFLiver->AddElement(Ca, 0.);
    AFLiver->AddElement(Fe, 0.);
    AFLiver->AddElement(I, 0.);

    rindexAFLiver = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFLiver);
    
    absLengthAFLiver = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFLiver.dat", absLengthAFLiver);
    
    G4MaterialPropertiesTable *mptAFLiver = DefineNonScintillatingMaterial(AFLiver, 13, rindexAFLiver, absLengthAFLiver);
    
    // Pancreas
    AFPancreas = new G4Material("AFPancreas", 1.05*g/cm3, 13);
    AFPancreas->AddElement(H, 0.105);
    AFPancreas->AddElement(C, 0.157);
    AFPancreas->AddElement(N, 0.024);
    AFPancreas->AddElement(O, 0.705);
    AFPancreas->AddElement(Na, 0.002);
    AFPancreas->AddElement(Mg, 0.);
    AFPancreas->AddElement(P, 0.002);
    AFPancreas->AddElement(S, 0.001);
    AFPancreas->AddElement(Cl, 0.002);
    AFPancreas->AddElement(K, 0.002);
    AFPancreas->AddElement(Ca, 0.);
    AFPancreas->AddElement(Fe, 0.);
    AFPancreas->AddElement(I, 0.);

    rindexAFPancreas = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFPancreas);
    
    absLengthAFPancreas = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFPancreas.dat", absLengthAFPancreas);
    
    G4MaterialPropertiesTable *mptAFPancreas = DefineNonScintillatingMaterial(AFPancreas, 13, rindexAFPancreas, absLengthAFPancreas);
    
    // Brain
    AFBrain = new G4Material("AFBrain", 1.05*g/cm3, 13);
    AFBrain->AddElement(H, 0.107);
    AFBrain->AddElement(C, 0.144);
    AFBrain->AddElement(N, 0.022);
    AFBrain->AddElement(O, 0.713);
    AFBrain->AddElement(Na, 0.002);
    AFBrain->AddElement(Mg, 0.);
    AFBrain->AddElement(P, 0.004);
    AFBrain->AddElement(S, 0.002);
    AFBrain->AddElement(Cl, 0.003);
    AFBrain->AddElement(K, 0.003);
    AFBrain->AddElement(Ca, 0.);
    AFBrain->AddElement(Fe, 0.);
    AFBrain->AddElement(I, 0.);

    rindexAFBrain = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFBrain);
    
    absLengthAFBrain = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFBrain.dat", absLengthAFBrain);
    
    G4MaterialPropertiesTable *mptAFBrain = DefineNonScintillatingMaterial(AFBrain, 13, rindexAFBrain, absLengthAFBrain);
    
    // Heart
    AFHeart = new G4Material("AFHeart", 1.05*g/cm3, 13);
    AFHeart->AddElement(H, 0.104);
    AFHeart->AddElement(C, 0.138);
    AFHeart->AddElement(N, 0.029);
    AFHeart->AddElement(O, 0.719);
    AFHeart->AddElement(Na, 0.001);
    AFHeart->AddElement(Mg, 0.);
    AFHeart->AddElement(P, 0.002);
    AFHeart->AddElement(S, 0.002);
    AFHeart->AddElement(Cl, 0.002);
    AFHeart->AddElement(K, 0.003);
    AFHeart->AddElement(Ca, 0.);
    AFHeart->AddElement(Fe, 0.);
    AFHeart->AddElement(I, 0.);

    rindexAFHeart = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFHeart);
    
    absLengthAFHeart = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFHeart.dat", absLengthAFHeart);
    
    G4MaterialPropertiesTable *mptAFHeart = DefineNonScintillatingMaterial(AFHeart, 13, rindexAFHeart, absLengthAFHeart);
    
    // Eyes
    AFEyes = new G4Material("AFEyes", 1.05*g/cm3, 13);
    AFEyes->AddElement(H, 0.097);
    AFEyes->AddElement(C, 0.183);
    AFEyes->AddElement(N, 0.054);
    AFEyes->AddElement(O, 0.66);
    AFEyes->AddElement(Na, 0.001);
    AFEyes->AddElement(Mg, 0.);
    AFEyes->AddElement(P, 0.001);
    AFEyes->AddElement(S, 0.003);
    AFEyes->AddElement(Cl, 0.001);
    AFEyes->AddElement(K, 0.);
    AFEyes->AddElement(Ca, 0.);
    AFEyes->AddElement(Fe, 0.);
    AFEyes->AddElement(I, 0.);

    rindexAFEyes = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFEyes);
    
    absLengthAFEyes = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFEyes.dat", absLengthAFEyes);
    
    G4MaterialPropertiesTable *mptAFEyes = DefineNonScintillatingMaterial(AFEyes, 13, rindexAFEyes, absLengthAFEyes);
    
    // Kidneys
    AFKidneys = new G4Material("AFKidneys", 1.05*g/cm3, 13);
    AFKidneys->AddElement(H, 0.103);
    AFKidneys->AddElement(C, 0.125);
    AFKidneys->AddElement(N, 0.031);
    AFKidneys->AddElement(O, 0.73);
    AFKidneys->AddElement(Na, 0.002);
    AFKidneys->AddElement(Mg, 0.);
    AFKidneys->AddElement(P, 0.002);
    AFKidneys->AddElement(S, 0.002);
    AFKidneys->AddElement(Cl, 0.002);
    AFKidneys->AddElement(K, 0.002);
    AFKidneys->AddElement(Ca, 0.001);
    AFKidneys->AddElement(Fe, 0.);
    AFKidneys->AddElement(I, 0.);

    rindexAFKidneys = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFKidneys);
    
    absLengthAFKidneys = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFKidneys.dat", absLengthAFKidneys);
    
    G4MaterialPropertiesTable *mptAFKidneys = DefineNonScintillatingMaterial(AFKidneys, 13, rindexAFKidneys, absLengthAFKidneys);
    
    // Stomach
    AFStomach = new G4Material("AFStomach", 1.04*g/cm3, 13);
    AFStomach->AddElement(H, 0.105);
    AFStomach->AddElement(C, 0.114);
    AFStomach->AddElement(N, 0.025);
    AFStomach->AddElement(O, 0.75);
    AFStomach->AddElement(Na, 0.001);
    AFStomach->AddElement(Mg, 0.);
    AFStomach->AddElement(P, 0.001);
    AFStomach->AddElement(S, 0.001);
    AFStomach->AddElement(Cl, 0.002);
    AFStomach->AddElement(K, 0.001);
    AFStomach->AddElement(Ca, 0.);
    AFStomach->AddElement(Fe, 0.);
    AFStomach->AddElement(I, 0.);

    rindexAFStomach = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFStomach);
    
    absLengthAFStomach = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFStomach.dat", absLengthAFStomach);
    
    G4MaterialPropertiesTable *mptAFStomach = DefineNonScintillatingMaterial(AFStomach, 13, rindexAFStomach, absLengthAFStomach);
    
    // SmallIntestine
    AFSmallIntestine = new G4Material("AFSmallIntestine", 1.04*g/cm3, 13);
    AFSmallIntestine->AddElement(H, 0.105);
    AFSmallIntestine->AddElement(C, 0.114);
    AFSmallIntestine->AddElement(N, 0.025);
    AFSmallIntestine->AddElement(O, 0.75);
    AFSmallIntestine->AddElement(Na, 0.001);
    AFSmallIntestine->AddElement(Mg, 0.);
    AFSmallIntestine->AddElement(P, 0.001);
    AFSmallIntestine->AddElement(S, 0.001);
    AFSmallIntestine->AddElement(Cl, 0.002);
    AFSmallIntestine->AddElement(K, 0.001);
    AFSmallIntestine->AddElement(Ca, 0.);
    AFSmallIntestine->AddElement(Fe, 0.);
    AFSmallIntestine->AddElement(I, 0.);

    rindexAFSmallIntestine = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFSmallIntestine);
    
    absLengthAFSmallIntestine = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFSmallIntestine.dat", absLengthAFSmallIntestine);
    
    G4MaterialPropertiesTable *mptAFSmallIntestine = DefineNonScintillatingMaterial(AFSmallIntestine, 13, rindexAFSmallIntestine, absLengthAFSmallIntestine);
    
    // LargeIntestine
    AFLargeIntestine = new G4Material("AFLargeIntestine", 1.04*g/cm3, 13);
    AFLargeIntestine->AddElement(H, 0.105);
    AFLargeIntestine->AddElement(C, 0.114);
    AFLargeIntestine->AddElement(N, 0.025);
    AFLargeIntestine->AddElement(O, 0.75);
    AFLargeIntestine->AddElement(Na, 0.001);
    AFLargeIntestine->AddElement(Mg, 0.);
    AFLargeIntestine->AddElement(P, 0.001);
    AFLargeIntestine->AddElement(S, 0.001);
    AFLargeIntestine->AddElement(Cl, 0.002);
    AFLargeIntestine->AddElement(K, 0.001);
    AFLargeIntestine->AddElement(Ca, 0.);
    AFLargeIntestine->AddElement(Fe, 0.);
    AFLargeIntestine->AddElement(I, 0.);

    rindexAFLargeIntestine = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFLargeIntestine);
    
    absLengthAFLargeIntestine = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFLargeIntestine.dat", absLengthAFLargeIntestine);
    
    G4MaterialPropertiesTable *mptAFLargeIntestine = DefineNonScintillatingMaterial(AFLargeIntestine, 13, rindexAFLargeIntestine, absLengthAFLargeIntestine);
    
    // Spleen
    AFSpleen = new G4Material("AFSpleen", 1.04*g/cm3, 13);
    AFSpleen->AddElement(H, 0.103);
    AFSpleen->AddElement(C, 0.112);
    AFSpleen->AddElement(N, 0.032);
    AFSpleen->AddElement(O, 0.743);
    AFSpleen->AddElement(Na, 0.001);
    AFSpleen->AddElement(Mg, 0.);
    AFSpleen->AddElement(P, 0.002);
    AFSpleen->AddElement(S, 0.002);
    AFSpleen->AddElement(Cl, 0.002);
    AFSpleen->AddElement(K, 0.003);
    AFSpleen->AddElement(Ca, 0.);
    AFSpleen->AddElement(Fe, 0.);
    AFSpleen->AddElement(I, 0.);

    rindexAFSpleen = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFSpleen);
    
    absLengthAFSpleen = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFSpleen.dat", absLengthAFSpleen);
    
    G4MaterialPropertiesTable *mptAFSpleen = DefineNonScintillatingMaterial(AFSpleen, 13, rindexAFSpleen, absLengthAFSpleen);
    
    // Thyroid
    AFThyroid = new G4Material("AFThyroid", 1.04*g/cm3, 13);
    AFThyroid->AddElement(H, 0.104);
    AFThyroid->AddElement(C, 0.118);
    AFThyroid->AddElement(N, 0.025);
    AFThyroid->AddElement(O, 0.745);
    AFThyroid->AddElement(Na, 0.002);
    AFThyroid->AddElement(Mg, 0.);
    AFThyroid->AddElement(P, 0.001);
    AFThyroid->AddElement(S, 0.001);
    AFThyroid->AddElement(Cl, 0.002);
    AFThyroid->AddElement(K, 0.001);
    AFThyroid->AddElement(Ca, 0.);
    AFThyroid->AddElement(Fe, 0.);
    AFThyroid->AddElement(I, 0.001);

    rindexAFThyroid = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFThyroid);
    
    absLengthAFThyroid = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFThyroid.dat", absLengthAFThyroid);
    
    G4MaterialPropertiesTable *mptAFThyroid = DefineNonScintillatingMaterial(AFThyroid, 13, rindexAFThyroid, absLengthAFThyroid);
    
    // UrinaryBladder
    AFUrinaryBladder = new G4Material("AFUrinaryBladder", 1.04*g/cm3, 13);
    AFUrinaryBladder->AddElement(H, 0.105);
    AFUrinaryBladder->AddElement(C, 0.096);
    AFUrinaryBladder->AddElement(N, 0.026);
    AFUrinaryBladder->AddElement(O, 0.761);
    AFUrinaryBladder->AddElement(Na, 0.002);
    AFUrinaryBladder->AddElement(Mg, 0.);
    AFUrinaryBladder->AddElement(P, 0.002);
    AFUrinaryBladder->AddElement(S, 0.002);
    AFUrinaryBladder->AddElement(Cl, 0.003);
    AFUrinaryBladder->AddElement(K, 0.003);
    AFUrinaryBladder->AddElement(Ca, 0.);
    AFUrinaryBladder->AddElement(Fe, 0.);
    AFUrinaryBladder->AddElement(I, 0.);

    rindexAFUrinaryBladder = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFUrinaryBladder);
    
    absLengthAFUrinaryBladder = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFUrinaryBladder.dat", absLengthAFUrinaryBladder);
    
    G4MaterialPropertiesTable *mptAFUrinaryBladder = DefineNonScintillatingMaterial(AFUrinaryBladder, 13, rindexAFUrinaryBladder, absLengthAFUrinaryBladder);
    
    // Ovaries
    AFOvaries = new G4Material("AFOvaries", 1.04*g/cm3, 13);
    AFOvaries->AddElement(H, 0.105);
    AFOvaries->AddElement(C, 0.094);
    AFOvaries->AddElement(N, 0.025);
    AFOvaries->AddElement(O, 0.766);
    AFOvaries->AddElement(Na, 0.002);
    AFOvaries->AddElement(Mg, 0.);
    AFOvaries->AddElement(P, 0.002);
    AFOvaries->AddElement(S, 0.002);
    AFOvaries->AddElement(Cl, 0.002);
    AFOvaries->AddElement(K, 0.002);
    AFOvaries->AddElement(Ca, 0.);
    AFOvaries->AddElement(Fe, 0.);
    AFOvaries->AddElement(I, 0.);

    rindexAFOvaries = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFOvaries);
    
    absLengthAFOvaries = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFOvaries.dat", absLengthAFOvaries);
    
    G4MaterialPropertiesTable *mptAFOvaries = DefineNonScintillatingMaterial(AFOvaries, 13, rindexAFOvaries, absLengthAFOvaries);
    
    // Adrenals
    AFAdrenals = new G4Material("AFAdrenals", 1.03*g/cm3, 13);
    AFAdrenals->AddElement(H, 0.104);
    AFAdrenals->AddElement(C, 0.228);
    AFAdrenals->AddElement(N, 0.028);
    AFAdrenals->AddElement(O, 0.63);
    AFAdrenals->AddElement(Na, 0.001);
    AFAdrenals->AddElement(Mg, 0.);
    AFAdrenals->AddElement(P, 0.002);
    AFAdrenals->AddElement(S, 0.003);
    AFAdrenals->AddElement(Cl, 0.002);
    AFAdrenals->AddElement(K, 0.002);
    AFAdrenals->AddElement(Ca, 0.);
    AFAdrenals->AddElement(Fe, 0.);
    AFAdrenals->AddElement(I, 0.);

    rindexAFAdrenals = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFAdrenals);
    
    absLengthAFAdrenals = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFAdrenals.dat", absLengthAFAdrenals);
    
    G4MaterialPropertiesTable *mptAFAdrenals = DefineNonScintillatingMaterial(AFAdrenals, 13, rindexAFAdrenals, absLengthAFAdrenals);
    
    // Oesophagus
    AFOesophagus = new G4Material("AFOesophagus", 1.03*g/cm3, 13);
    AFOesophagus->AddElement(H, 0.104);
    AFOesophagus->AddElement(C, 0.222);
    AFOesophagus->AddElement(N, 0.028);
    AFOesophagus->AddElement(O, 0.636);
    AFOesophagus->AddElement(Na, 0.001);
    AFOesophagus->AddElement(Mg, 0.);
    AFOesophagus->AddElement(P, 0.002);
    AFOesophagus->AddElement(S, 0.003);
    AFOesophagus->AddElement(Cl, 0.002);
    AFOesophagus->AddElement(K, 0.002);
    AFOesophagus->AddElement(Ca, 0.);
    AFOesophagus->AddElement(Fe, 0.);
    AFOesophagus->AddElement(I, 0.);

    rindexAFOesophagus = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFOesophagus);
    
    absLengthAFOesophagus = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFOesophagus.dat", absLengthAFOesophagus);
    
    G4MaterialPropertiesTable *mptAFOesophagus = DefineNonScintillatingMaterial(AFOesophagus, 13, rindexAFOesophagus, absLengthAFOesophagus);
    
    // Gallbladder
    AFGallbladder = new G4Material("AFGallbladder", 1.03*g/cm3, 13);
    AFGallbladder->AddElement(H, 0.105);
    AFGallbladder->AddElement(C, 0.235);
    AFGallbladder->AddElement(N, 0.028);
    AFGallbladder->AddElement(O, 0.622);
    AFGallbladder->AddElement(Na, 0.001);
    AFGallbladder->AddElement(Mg, 0.);
    AFGallbladder->AddElement(P, 0.002);
    AFGallbladder->AddElement(S, 0.003);
    AFGallbladder->AddElement(Cl, 0.002);
    AFGallbladder->AddElement(K, 0.002);
    AFGallbladder->AddElement(Ca, 0.);
    AFGallbladder->AddElement(Fe, 0.);
    AFGallbladder->AddElement(I, 0.);

    rindexAFGallbladder = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFGallbladder);
    
    absLengthAFGallbladder = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFGallbladder.dat", absLengthAFGallbladder);
    
    G4MaterialPropertiesTable *mptAFGallbladder = DefineNonScintillatingMaterial(AFGallbladder, 13, rindexAFGallbladder, absLengthAFGallbladder);
    
    // Uterus
    AFUterus = new G4Material("AFUterus", 1.03*g/cm3, 13);
    AFUterus->AddElement(H, 0.105);
    AFUterus->AddElement(C, 0.286);
    AFUterus->AddElement(N, 0.025);
    AFUterus->AddElement(O, 0.576);
    AFUterus->AddElement(Na, 0.001);
    AFUterus->AddElement(Mg, 0.);
    AFUterus->AddElement(P, 0.002);
    AFUterus->AddElement(S, 0.002);
    AFUterus->AddElement(Cl, 0.001);
    AFUterus->AddElement(K, 0.002);
    AFUterus->AddElement(Ca, 0.);
    AFUterus->AddElement(Fe, 0.);
    AFUterus->AddElement(I, 0.);

    rindexAFUterus = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFUterus);
    
    absLengthAFUterus = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFUterus.dat", absLengthAFUterus);
    
    G4MaterialPropertiesTable *mptAFUterus = DefineNonScintillatingMaterial(AFUterus, 13, rindexAFUterus, absLengthAFUterus);
    
    // Lymph
    AFLymph = new G4Material("AFLymph", 1.03*g/cm3, 13);
    AFLymph->AddElement(H, 0.108);
    AFLymph->AddElement(C, 0.042);
    AFLymph->AddElement(N, 0.011);
    AFLymph->AddElement(O, 0.831);
    AFLymph->AddElement(Na, 0.003);
    AFLymph->AddElement(Mg, 0.);
    AFLymph->AddElement(P, 0.);
    AFLymph->AddElement(S, 0.001);
    AFLymph->AddElement(Cl, 0.004);
    AFLymph->AddElement(K, 0.);
    AFLymph->AddElement(Ca, 0.);
    AFLymph->AddElement(Fe, 0.);
    AFLymph->AddElement(I, 0.);

    rindexAFLymph = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFLymph);
    
    absLengthAFLymph = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFLymph.dat", absLengthAFLymph);
    
    G4MaterialPropertiesTable *mptAFLymph = DefineNonScintillatingMaterial(AFLymph, 13, rindexAFLymph, absLengthAFLymph);
    
    // Breast
    AFBreast = new G4Material("AFBreast", 1.02*g/cm3, 13);
    AFBreast->AddElement(H, 0.114);
    AFBreast->AddElement(C, 0.461);
    AFBreast->AddElement(N, 0.005);
    AFBreast->AddElement(O, 0.42);
    AFBreast->AddElement(Na, 0.);
    AFBreast->AddElement(Mg, 0.);
    AFBreast->AddElement(P, 0.);
    AFBreast->AddElement(S, 0.);
    AFBreast->AddElement(Cl, 0.);
    AFBreast->AddElement(K, 0.);
    AFBreast->AddElement(Ca, 0.);
    AFBreast->AddElement(Fe, 0.);
    AFBreast->AddElement(I, 0.);

    rindexAFBreast = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFBreast);
    
    absLengthAFBreast = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFBreast.dat", absLengthAFBreast);
    
    G4MaterialPropertiesTable *mptAFBreast = DefineNonScintillatingMaterial(AFBreast, 13, rindexAFBreast, absLengthAFBreast);
    
    // AdiposeTissue
    AFAdiposeTissue = new G4Material("AFAdiposeTissue", 0.95*g/cm3, 13);
    AFAdiposeTissue->AddElement(H, 0.114);
    AFAdiposeTissue->AddElement(C, 0.589);
    AFAdiposeTissue->AddElement(N, 0.007);
    AFAdiposeTissue->AddElement(O, 0.287);
    AFAdiposeTissue->AddElement(Na, 0.001);
    AFAdiposeTissue->AddElement(Mg, 0.);
    AFAdiposeTissue->AddElement(P, 0.);
    AFAdiposeTissue->AddElement(S, 0.001);
    AFAdiposeTissue->AddElement(Cl, 0.001);
    AFAdiposeTissue->AddElement(K, 0.);
    AFAdiposeTissue->AddElement(Ca, 0.);
    AFAdiposeTissue->AddElement(Fe, 0.);
    AFAdiposeTissue->AddElement(I, 0.);

    rindexAFAdiposeTissue = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFAdiposeTissue);
    
    absLengthAFAdiposeTissue = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFAdiposeTissue.dat", absLengthAFAdiposeTissue);
    
    G4MaterialPropertiesTable *mptAFAdiposeTissue = DefineNonScintillatingMaterial(AFAdiposeTissue, 13, rindexAFAdiposeTissue, absLengthAFAdiposeTissue);
    
    // Lung
    AFLung = new G4Material("AFLung", 0.385*g/cm3, 13);
    AFLung->AddElement(H, 0.103);
    AFLung->AddElement(C, 0.107);
    AFLung->AddElement(N, 0.032);
    AFLung->AddElement(O, 0.746);
    AFLung->AddElement(Na, 0.002);
    AFLung->AddElement(Mg, 0.);
    AFLung->AddElement(P, 0.002);
    AFLung->AddElement(S, 0.003);
    AFLung->AddElement(Cl, 0.003);
    AFLung->AddElement(K, 0.002);
    AFLung->AddElement(Ca, 0.);
    AFLung->AddElement(Fe, 0.);
    AFLung->AddElement(I, 0.);

    rindexAFLung = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFLung);
    
    absLengthAFLung = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFLung.dat", absLengthAFLung);
    
    G4MaterialPropertiesTable *mptAFLung = DefineNonScintillatingMaterial(AFLung, 13, rindexAFLung, absLengthAFLung);
    
    // GastroIntestinalContents
    AFGastroIntestinalContents = new G4Material("AFGastroIntestinalContents", 1.04*g/cm3, 13);
    AFGastroIntestinalContents->AddElement(H, 0.1);
    AFGastroIntestinalContents->AddElement(C, 0.222);
    AFGastroIntestinalContents->AddElement(N, 0.022);
    AFGastroIntestinalContents->AddElement(O, 0.644);
    AFGastroIntestinalContents->AddElement(Na, 0.001);
    AFGastroIntestinalContents->AddElement(Mg, 0.);
    AFGastroIntestinalContents->AddElement(P, 0.002);
    AFGastroIntestinalContents->AddElement(S, 0.003);
    AFGastroIntestinalContents->AddElement(Cl, 0.001);
    AFGastroIntestinalContents->AddElement(K, 0.004);
    AFGastroIntestinalContents->AddElement(Ca, 0.001);
    AFGastroIntestinalContents->AddElement(Fe, 0.);
    AFGastroIntestinalContents->AddElement(I, 0.);

    rindexAFGastroIntestinalContents = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFGastroIntestinalContents);
    
    absLengthAFGastroIntestinalContents = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFGastroIntestinalContents.dat", absLengthAFGastroIntestinalContents);
    
    G4MaterialPropertiesTable *mptAFGastroIntestinalContents = DefineNonScintillatingMaterial(AFGastroIntestinalContents, 13, rindexAFGastroIntestinalContents, absLengthAFGastroIntestinalContents);
    
    // Urine
    AFUrine = new G4Material("AFUrine", 1.04*g/cm3, 13);
    AFUrine->AddElement(H, 0.107);
    AFUrine->AddElement(C, 0.003);
    AFUrine->AddElement(N, 0.01);
    AFUrine->AddElement(O, 0.873);
    AFUrine->AddElement(Na, 0.004);
    AFUrine->AddElement(Mg, 0.);
    AFUrine->AddElement(P, 0.001);
    AFUrine->AddElement(S, 0.);
    AFUrine->AddElement(Cl, 0.);
    AFUrine->AddElement(K, 0.002);
    AFUrine->AddElement(Ca, 0.);
    AFUrine->AddElement(Fe, 0.);
    AFUrine->AddElement(I, 0.);

    rindexAFUrine = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFUrine);
    
    absLengthAFUrine = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFUrine.dat", absLengthAFUrine);
    
    G4MaterialPropertiesTable *mptAFUrine = DefineNonScintillatingMaterial(AFUrine, 13, rindexAFUrine, absLengthAFUrine);
    
    // Air
    AFAir = new G4Material("AFAir", 0.001*g/cm3, 13);
    AFAir->AddElement(H, 0.);
    AFAir->AddElement(C, 0.);
    AFAir->AddElement(N, 0.8);
    AFAir->AddElement(O, 0.2);
    AFAir->AddElement(Na, 0.);
    AFAir->AddElement(Mg, 0.);
    AFAir->AddElement(P, 0.);
    AFAir->AddElement(S, 0.);
    AFAir->AddElement(Cl, 0.);
    AFAir->AddElement(K, 0.);
    AFAir->AddElement(Ca, 0.);
    AFAir->AddElement(Fe, 0.);
    AFAir->AddElement(I, 0.);

    rindexAFAir = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFAir);
    
    absLengthAFAir = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFAir.dat", absLengthAFAir);
    
    G4MaterialPropertiesTable *mptAFAir = DefineNonScintillatingMaterial(AFAir, 13, rindexAFAir, absLengthAFAir);
    
    // IodinatedBlood
    AFIodinatedBlood = new G4Material("AFIodinatedBlood", 1.06*g/cm3, 13);
    AFIodinatedBlood->AddElement(H, 0.10098);
    AFIodinatedBlood->AddElement(C, 0.1089);
    AFIodinatedBlood->AddElement(N, 0.03267);
    AFIodinatedBlood->AddElement(O, 0.73755);
    AFIodinatedBlood->AddElement(Na, 0.00099);
    AFIodinatedBlood->AddElement(Mg, 0.);
    AFIodinatedBlood->AddElement(P, 0.00099);
    AFIodinatedBlood->AddElement(S, 0.00198);
    AFIodinatedBlood->AddElement(Cl, 0.00297);
    AFIodinatedBlood->AddElement(K, 0.00198);
    AFIodinatedBlood->AddElement(Ca, 0.);
    AFIodinatedBlood->AddElement(Fe, 0.00099);
    AFIodinatedBlood->AddElement(I, 0.01);

    rindexAFIodinatedBlood = new G4PhysicsOrderedFreeVector();
    ReadDataFile("rindexAir.dat", rindexAFIodinatedBlood);
    
    absLengthAFIodinatedBlood = new G4PhysicsOrderedFreeVector();
    ReadDataFile("linAttCoeffAFIodinatedBlood.dat", absLengthAFIodinatedBlood);
    
    G4MaterialPropertiesTable *mptAFIodinatedBlood = DefineNonScintillatingMaterial(AFIodinatedBlood, 13, rindexAFIodinatedBlood, absLengthAFIodinatedBlood);
}

void NSDetectorConstruction::DefineMaterials()
{
    G4NistManager *nist = G4NistManager::Instance();
    
    DefineElements(nist);
    DefineWorld(nist);
    DefineDetector(nist);
    
    DefineYAGCe();
    DefineZnSeTe();
    DefineBaF2();
    DefineLaCl3Ce();
    DefineLYSOCe();
    DefineCsITl();
    DefineGSOCe();
    DefineNaITl();
    DefineGadoxTb();
    DefineSiO2();
    DefineTiO2();
    
    if (sampleID == 1)
    {
        DefineAMBioMedia();
    } else if (sampleID == 2)
    {
        DefineAFBioMedia();
    } else if (sampleID == 3)
    {
        DefineAMBioMedia();
    }
}

void NSDetectorConstruction::ConstructCylindricalPhantom()
{
    const G4double zOffset = 0.6*rWorld*cm;
    
    solidSoftTissue = new G4Box("solidSoftTissue", xSample/2*cm, ySample/2*cm, zSample/2*cm);
    logicSoftTissue = new G4LogicalVolume(solidSoftTissue, AMAdiposeTissue, "logicSoftTissue");
    physSoftTissue = new G4PVPlacement(0, G4ThreeVector(0.0, 0.0, -zOffset-zSample/2*cm), logicSoftTissue, "physSoftTissue", logicWorld, false, 0, checkDetectorsOverlaps);
    fSampleVolumes.push_back(logicSoftTissue);

    solidMuscle = new G4Tubs("solidMuscle", 0.0, 0.175*xSample*cm, zSample/2*cm, 0.0*deg, 360.0*deg);
    logicMuscle = new G4LogicalVolume(solidMuscle, AMMuscle, "logicMuscle");
    physMuscle = new G4PVPlacement(0, G4ThreeVector(0.225*xSample*cm, -0.225*xSample*cm, 0.0), logicMuscle, "physMuscle", logicSoftTissue, false, 0, checkDetectorsOverlaps);
    fSampleVolumes.push_back(logicMuscle);
    
    solidBone = new G4Tubs("solidBone", 0.0, 0.175*xSample*cm, 0.05*zSample/2*cm, 0.0*deg, 360.0*deg);
    logicBone = new G4LogicalVolume(solidBone, AMMineralBone, "logicBone");
    physBone = new G4PVPlacement(0, G4ThreeVector(-0.225*xSample*cm, -0.225*xSample*cm, 0.0), logicBone, "physBone", logicSoftTissue, false, 0, checkDetectorsOverlaps);
    fSampleVolumes.push_back(logicBone);
    
    solidIBlood = new G4Tubs("solidIBlood", 0.0, 0.175*xSample*cm, 0.35*zSample/2*cm, 0.0*deg, 360.0*deg);
    logicIBlood = new G4LogicalVolume(solidIBlood, AMIBlood, "logicIBlood");
    physIBlood = new G4PVPlacement(0, G4ThreeVector(-0.225*xSample*cm, 0.225*xSample*cm, 0.0), logicIBlood, "physIBlood", logicSoftTissue, false, 0, checkDetectorsOverlaps);
    fSampleVolumes.push_back(logicIBlood);
    
    solidGdBlood = new G4Tubs("solidGdBlood", 0.0, 0.175*xSample*cm, 0.5*zSample/2*cm, 0.0*deg, 360.0*deg);
    logicGdBlood = new G4LogicalVolume(solidGdBlood, AMGdBlood, "logicGdBlood");
    physGdBlood = new G4PVPlacement(0, G4ThreeVector(0.225*xSample*cm, 0.225*xSample*cm, 0.0), logicGdBlood, "physGdBlood", logicSoftTissue, false, 0, checkDetectorsOverlaps);
    fSampleVolumes.push_back(logicGdBlood);
}

void NSDetectorConstruction::ConstructSample()
{
    const G4double voxelX = xSample/nSampleX*cm, voxelY = ySample/nSampleY*cm, voxelZ = zSample/nSampleZ*cm;
    const G4double zOffset = 0.6*rWorld*cm;
    solidSample = new G4Box("solidSample", voxelX/2, voxelY/2, voxelZ/2);
    
    if (sampleID == 1)
    {
        sampleName = "AM";
    } else if (sampleID == 2)
    {
        sampleName = "AF";
    }
    
    for(G4int k = indSampleZ; k < indSampleZ+nSampleZ; k++)
    {
        std::ifstream is(sampleName+"slice"+std::to_string(k)+".dat");
        std::istream_iterator<int> start(is), end;
        std::vector<int> organID(start, end);
        
        for(G4int j = indSampleY; j < indSampleY+nSampleY; j++)
        {
            for(G4int i = indSampleX; i < indSampleX+nSampleX; i++)
            {
                G4int organIDij = organID[i+nSampleXall*j];
                if (sampleID == 1)
                {
                    if (organIDij == 128)
                    {
                        sampleMat = AMTeeth;
                    } else if (organIDij == 13 || organIDij == 16 || organIDij == 19 || organIDij == 22 || organIDij == 24 || organIDij == 26 || organIDij == 28 || organIDij == 31 || organIDij == 34 || organIDij == 37 || organIDij == 39 || organIDij == 41 || organIDij == 43 || organIDij == 45 || organIDij == 47 || organIDij == 49 || organIDij == 51 || organIDij == 53 || organIDij == 55)
                    {
                        sampleMat = AMMineralBone;
                    } else if (organIDij == 14)
                    {
                        sampleMat = AMHumeriUpper;
                    } else if (organIDij == 17)
                    {
                        sampleMat = AMHumeriLower;
                    } else if (organIDij == 20)
                    {
                        sampleMat = AMLowerArmBone;
                    } else if (organIDij == 23)
                    {
                        sampleMat = AMHandBone;
                    } else if (organIDij == 25)
                    {
                        sampleMat = AMClavicles;
                    } else if (organIDij == 27)
                    {
                        sampleMat = AMCranium;
                    } else if (organIDij == 29)
                    {
                        sampleMat = AMFemoraUpper;
                    } else if (organIDij == 32)
                    {
                        sampleMat = AMFemoraLower;
                    } else if (organIDij == 35)
                    {
                        sampleMat = AMLowerLeg;
                    } else if (organIDij == 38)
                    {
                        sampleMat = AMFoot;
                    } else if (organIDij == 40)
                    {
                        sampleMat = AMMandible;
                    } else if (organIDij == 42)
                    {
                        sampleMat = AMPelvis;
                    } else if (organIDij == 44)
                    {
                        sampleMat = AMRibs;
                    } else if (organIDij == 46)
                    {
                        sampleMat = AMScapulae;
                    } else if (organIDij == 48)
                    {
                        sampleMat = AMCervicalSpine;
                    } else if (organIDij == 50)
                    {
                        sampleMat = AMThoracicSpine;
                    } else if (organIDij == 52)
                    {
                        sampleMat = AMLumbarSpine;
                    } else if (organIDij == 54)
                    {
                        sampleMat = AMSacrum;
                    } else if (organIDij == 56)
                    {
                        sampleMat = AMSternum;
                    } else if (organIDij == 15 || organIDij == 30)
                    {
                        sampleMat = AMHumeriFemoraUpperMedullaryCavity;
                    } else if (organIDij == 18 || organIDij == 33)
                    {
                        sampleMat = AMHumeriFemoraLowerMedullaryCavity;
                    } else if (organIDij == 21)
                    {
                        sampleMat = AMLowerArmMedullaryCavity;
                    } else if (organIDij == 36)
                    {
                        sampleMat = AMLowerLegMedullaryCavity;
                    } else if (organIDij == 57 || organIDij == 58 || organIDij == 59 || organIDij == 60)
                    {
                        sampleMat = AMCartilage;
                    } else if (organIDij == 122 || organIDij == 123 || organIDij == 124 || organIDij == 125)
                    {
                        sampleMat = AMSkin;
                    } else if (organIDij == 9 || organIDij == 11 || organIDij == 12 || organIDij == 96 || organIDij == 98)
                    {
                        sampleMat = AMBlood;
                    } else if (organIDij == 10)
                    {
                        sampleMat = AMIBlood;
                    } else if (organIDij == 88)
                    {
                        sampleMat = AMGdBlood;
                    } else if (organIDij == 5 || organIDij == 6 || organIDij == 106 || organIDij == 107 || organIDij == 108 || organIDij == 109 || organIDij == 133)
                    {
                        sampleMat = AMMuscle;
                    } else if (organIDij == 95)
                    {
                        sampleMat = AMLiver;
                    } else if (organIDij == 113)
                    {
                        sampleMat = AMPancreas;
                    } else if (organIDij == 61)
                    {
                        sampleMat = AMBrain;
                    } else if (organIDij == 87)
                    {
                        sampleMat = AMHeart;
                    } else if (organIDij == 66 || organIDij == 67 || organIDij == 68 || organIDij == 69)
                    {
                        sampleMat = AMEyes;
                    } else if (organIDij == 89 || organIDij == 90 || organIDij == 91 || organIDij == 92 || organIDij == 93 || organIDij == 94)
                    {
                        sampleMat = AMKidneys;
                    } else if (organIDij == 72)
                    {
                        sampleMat = AMStomach;
                    } else if (organIDij == 74)
                    {
                        sampleMat = AMSmallIntestine;
                    } else if (organIDij == 76 || organIDij == 78 || organIDij == 80 || organIDij == 82 || organIDij == 84 || organIDij == 86)
                    {
                        sampleMat = AMLargeIntestine;
                    } else if (organIDij == 127)
                    {
                        sampleMat = AMSpleen;
                    } else if (organIDij == 132)
                    {
                        sampleMat = AMThyroid;
                    } else if (organIDij == 137)
                    {
                        sampleMat = AMUrinaryBladder;
                    } else if (organIDij == 111 || organIDij == 112 || organIDij == 129 || organIDij == 130)
                    {
                        sampleMat = AMTestes;
                    } else if (organIDij == 1 || organIDij == 2)
                    {
                        sampleMat = AMAdrenals;
                    } else if (organIDij == 110)
                    {
                        sampleMat = AMOesophagus;
                    } else if (organIDij == 3 || organIDij == 4 || organIDij == 7 || organIDij == 8 || organIDij == 70 || organIDij == 71 || organIDij == 114 || organIDij == 120 || organIDij == 121 || organIDij == 126 || organIDij == 131 || organIDij == 134 || organIDij == 135 || organIDij == 136)
                    {
                        sampleMat = AMGallbladder;
                    } else if (organIDij == 115 || organIDij == 139)
                    {
                        sampleMat = AMProstate;
                    } else if (organIDij == 100 || organIDij == 101 || organIDij == 102 || organIDij == 103 || organIDij == 104 || organIDij == 105)
                    {
                        sampleMat = AMLymph;
                    } else if (organIDij == 63 || organIDij == 65)
                    {
                        sampleMat = AMBreast;
                    } else if (organIDij == 62 || organIDij == 64 || organIDij == 116 || organIDij == 117 || organIDij == 118 || organIDij == 119)
                    {
                        sampleMat = AMAdiposeTissue;
                    } else if (organIDij == 97 || organIDij == 99)
                    {
                        sampleMat = AMLung;
                    } else if (organIDij == 73 || organIDij == 75 || organIDij == 77 || organIDij == 79 || organIDij == 81 || organIDij == 83 || organIDij == 85)
                    {
                        sampleMat = AMGastroIntestinalContents;
                    } else if (organIDij == 138)
                    {
                        sampleMat = AMUrine;
                    } else if (organIDij == 140 || organIDij == 0)
                    {
                        sampleMat = AMAir;
                    }
                } else if (sampleID == 2)
                {
                    if (organIDij == 128)
                    {
                        sampleMat = AFTeeth;
                    } else if (organIDij == 13 || organIDij == 16 || organIDij == 19 || organIDij == 22 || organIDij == 24 || organIDij == 26 || organIDij == 28 || organIDij == 31 || organIDij == 34 || organIDij == 37 || organIDij == 39 || organIDij == 41 || organIDij == 43 || organIDij == 45 || organIDij == 47 || organIDij == 49 || organIDij == 51 || organIDij == 53 || organIDij == 55)
                    {
                        sampleMat = AFMineralBone;
                    } else if (organIDij == 14)
                    {
                        sampleMat = AFHumeriUpper;
                    } else if (organIDij == 17)
                    {
                        sampleMat = AFHumeriLower;
                    } else if (organIDij == 20)
                    {
                        sampleMat = AFLowerArmBone;
                    } else if (organIDij == 23)
                    {
                        sampleMat = AFHandBone;
                    } else if (organIDij == 25)
                    {
                        sampleMat = AFClavicles;
                    } else if (organIDij == 27)
                    {
                        sampleMat = AFCranium;
                    } else if (organIDij == 29)
                    {
                        sampleMat = AFFemoraUpper;
                    } else if (organIDij == 32)
                    {
                        sampleMat = AFFemoraLower;
                    } else if (organIDij == 35)
                    {
                        sampleMat = AFLowerLeg;
                    } else if (organIDij == 38)
                    {
                        sampleMat = AFFoot;
                    } else if (organIDij == 40)
                    {
                        sampleMat = AFMandible;
                    } else if (organIDij == 42)
                    {
                        sampleMat = AFPelvis;
                    } else if (organIDij == 44)
                    {
                        sampleMat = AFRibs;
                    } else if (organIDij == 46)
                    {
                        sampleMat = AFScapulae;
                    } else if (organIDij == 48)
                    {
                        sampleMat = AFCervicalSpine;
                    } else if (organIDij == 50)
                    {
                        sampleMat = AFThoracicSpine;
                    } else if (organIDij == 52)
                    {
                        sampleMat = AFLumbarSpine;
                    } else if (organIDij == 54)
                    {
                        sampleMat = AFSacrum;
                    } else if (organIDij == 56)
                    {
                        sampleMat = AFSternum;
                    } else if (organIDij == 15 || organIDij == 30)
                    {
                        sampleMat = AFHumeriFemoraUpperMedullaryCavity;
                    } else if (organIDij == 18 || organIDij == 33)
                    {
                        sampleMat = AFHumeriFemoraLowerMedullaryCavity;
                    } else if (organIDij == 21)
                    {
                        sampleMat = AFLowerArmMedullaryCavity;
                    } else if (organIDij == 36)
                    {
                        sampleMat = AFLowerLegMedullaryCavity;
                    } else if (organIDij == 57 || organIDij == 58 || organIDij == 59 || organIDij == 60)
                    {
                        sampleMat = AFCartilage;
                    } else if (organIDij == 122 || organIDij == 123 || organIDij == 124 || organIDij == 125)
                    {
                        sampleMat = AFSkin;
                    } else if (organIDij == 9 || organIDij == 10 || organIDij == 11 || organIDij == 12 || organIDij == 88 || organIDij == 96 || organIDij == 98)
                    {
                        sampleMat = AFBlood;
                    } else if (organIDij == 5 || organIDij == 6 || organIDij == 106 || organIDij == 107 || organIDij == 108 || organIDij == 109 || organIDij == 133)
                    {
                        sampleMat = AFMuscle;
                    } else if (organIDij == 95)
                    {
                        sampleMat = AFLiver;
                    } else if (organIDij == 113)
                    {
                        sampleMat = AFPancreas;
                    } else if (organIDij == 61)
                    {
                        sampleMat = AFBrain;
                    } else if (organIDij == 87)
                    {
                        sampleMat = AFHeart;
                    } else if (organIDij == 66 || organIDij == 67 || organIDij == 68 || organIDij == 69)
                    {
                        sampleMat = AFEyes;
                    } else if (organIDij == 89 || organIDij == 90 || organIDij == 91 || organIDij == 92 || organIDij == 93 || organIDij == 94)
                    {
                        sampleMat = AFKidneys;
                    } else if (organIDij == 72)
                    {
                        sampleMat = AFStomach;
                    } else if (organIDij == 74)
                    {
                        sampleMat = AFSmallIntestine;
                    } else if (organIDij == 76 || organIDij == 78 || organIDij == 80 || organIDij == 82 || organIDij == 84 || organIDij == 86)
                    {
                        sampleMat = AFLargeIntestine;
                    } else if (organIDij == 127)
                    {
                        sampleMat = AFSpleen;
                    } else if (organIDij == 132)
                    {
                        sampleMat = AFThyroid;
                    } else if (organIDij == 137)
                    {
                        sampleMat = AFUrinaryBladder;
                    } else if (organIDij == 111 || organIDij == 112 || organIDij == 129 || organIDij == 130)
                    {
                        sampleMat = AFOvaries;
                    } else if (organIDij == 1 || organIDij == 2)
                    {
                        sampleMat = AFAdrenals;
                    } else if (organIDij == 110)
                    {
                        sampleMat = AFOesophagus;
                    } else if (organIDij == 3 || organIDij == 4 || organIDij == 7 || organIDij == 8 || organIDij == 70 || organIDij == 71 || organIDij == 114 || organIDij == 120 || organIDij == 121 || organIDij == 126 || organIDij == 131 || organIDij == 134 || organIDij == 135 || organIDij == 136)
                    {
                        sampleMat = AFGallbladder;
                    } else if (organIDij == 115 || organIDij == 139)
                    {
                        sampleMat = AFUterus;
                    } else if (organIDij == 100 || organIDij == 101 || organIDij == 102 || organIDij == 103 || organIDij == 104 || organIDij == 105)
                    {
                        sampleMat = AFLymph;
                    } else if (organIDij == 63 || organIDij == 65)
                    {
                        sampleMat = AFBreast;
                    } else if (organIDij == 62 || organIDij == 64 || organIDij == 116 || organIDij == 117 || organIDij == 118 || organIDij == 119)
                    {
                        sampleMat = AFAdiposeTissue;
                    } else if (organIDij == 97 || organIDij == 99)
                    {
                        sampleMat = AFLung;
                    } else if (organIDij == 73 || organIDij == 75 || organIDij == 77 || organIDij == 79 || organIDij == 81 || organIDij == 83 || organIDij == 85)
                    {
                        sampleMat = AFGastroIntestinalContents;
                    } else if (organIDij == 138)
                    {
                        sampleMat = AFUrine;
                    } else if (organIDij == 140 || organIDij == 0)
                    {
                        sampleMat = AFAir;
                    }
                }
                logicSample = new G4LogicalVolume(solidSample, sampleMat, "logicSample");
                physSample = new G4PVPlacement(0, G4ThreeVector(-xSample/2*cm+(i-indSampleX+0.5)*voxelX, -ySample/2*cm+(j-indSampleY+0.5)*voxelY, -zOffset+(k-indSampleZ+0.5)*voxelZ), logicSample, "physSample", logicWorld, false, (i-indSampleX)+(j-indSampleY)*nSampleX+(k-indSampleZ)*nSampleX*nSampleY, checkDetectorsOverlaps);
                fSampleVolumes.push_back(logicSample);
            }
        }
    }
}

void NSDetectorConstruction::ConstructScintillators()
{
    const G4double zOffset = 0.6*rWorld*cm + zSample*cm + gapSampleScint*cm;

    if (materialScintCorr == 1)
        matNameScintCorr = YAGCe;
    else if (materialScintCorr == 2)
        matNameScintCorr = ZnSeTe;
    else if (materialScintCorr == 3)
        matNameScintCorr = LYSOCe;
    else if (materialScintCorr == 4)
        matNameScintCorr = CsITl;
    else if (materialScintCorr == 5)
        matNameScintCorr = GSOCe;
    else if (materialScintCorr == 6)
        matNameScintCorr = NaITl;
    else if (materialScintCorr == 7)
        matNameScintCorr = GadoxTb;

    solidScintCorr = new G4Tubs("solidScintCorr", 0.0, rScintCorr*cm, zScintCorr/2*cm, 0.0*deg, 360.0*deg);
    logicScintCorr = new G4LogicalVolume(solidScintCorr, matNameScintCorr, "logicScintCorr");
   	rotScintCorr = new G4RotationMatrix();
	rotScintCorr -> rotateY(-90.0*deg);
    G4ThreeVector transScintCorr = G4ThreeVector(0., 0., 0.);
    physScintCorr = new G4PVPlacement(rotScintCorr, transScintCorr, logicScintCorr, "physScintCorr", logicWorld, false, 0, checkDetectorsOverlaps);

    G4LogicalBorderSurface *interfaceScintCorr = new G4LogicalBorderSurface("interfaceScintCorr", physWorld, physScintCorr, opticalSurfaceWorld);

    if (materialScintImg == 1)
        matNameScintImg = YAGCe;
    else if (materialScintImg == 2)
        matNameScintImg = ZnSeTe;
    else if (materialScintImg == 3)
        matNameScintImg = LYSOCe;
    else if (materialScintImg == 4)
        matNameScintImg = CsITl;
    else if (materialScintImg == 5)
        matNameScintImg = GSOCe;
    else if (materialScintImg == 6)
        matNameScintImg = NaITl;
    else if (materialScintImg == 7)
        matNameScintImg = GadoxTb;

    solidScintImg = new G4Box("solidScintImg", xScintImg/2*cm, yScintImg/2*cm, zScintImg/2*cm);
    logicScintImg = new G4LogicalVolume(solidScintImg, matNameScintImg, "logicScintImg");
    G4ThreeVector transScintImg = G4ThreeVector(0., 0., -zOffset+zScintImg/2*cm);
    physScintImg = new G4PVPlacement(0, transScintImg, logicScintImg, "physScintImg", logicWorld, false, 0, checkDetectorsOverlaps);

    G4LogicalBorderSurface *interfaceScintImg = new G4LogicalBorderSurface("interfaceScintImg", physWorld, physScintImg, opticalSurfaceWorld);
}

void NSDetectorConstruction::ConstructSensitiveDetector()
{
    // Constructing the correlation detector
    const G4double detectorX = xDet/nDetX*um, detectorY = yDet/nDetY*um;
    const G4double zOffsetCorr = rScintCorr*cm + gapScintDet*cm, zOffsetImg = 0.6*rWorld*cm + zSample*cm + gapSampleScint*cm + zScintImg*cm + gapScintDet*cm;
    solidDetector = new G4Box("solidDetector", detectorX/2, detectorY/2, detectorDepth/2*um);
    logicDetector = new G4LogicalVolume(solidDetector, worldMat, "logicDetector");
    logicDetector->SetUserLimits(userLimits);
    NSSensitiveDetector *sensDet = new NSSensitiveDetector("SensitiveDetector", true);
    logicDetector->SetSensitiveDetector(sensDet);
    for(G4int i = 0; i < nDetX; i++)
    {
        for(G4int j = 0; j < nDetY; j++)
        {
            G4ThreeVector transDetCorr = G4ThreeVector(-xDet/2*um+(i+0.5)*detectorX, -yDet/2*um+(j+0.5)*detectorY, -zOffsetCorr+detectorDepth/2*um);
            G4ThreeVector transDetImg = G4ThreeVector(-xDet/2*um+(i+0.5)*detectorX, -yDet/2*um+(j+0.5)*detectorY, -zOffsetImg+detectorDepth/2*um);
            physDetector = new G4PVPlacement(0, transDetCorr, logicDetector, "physDetector", logicWorld, false, j+i*nDetY, checkDetectorsOverlaps);
            physDetector = new G4PVPlacement(0, transDetImg, logicDetector, "physDetector", logicWorld, false, j+i*nDetY+nDetX*nDetY, checkDetectorsOverlaps);
        }
    }
}

G4VPhysicalVolume *NSDetectorConstruction::Construct()
{
    DefineMaterials();
    
    navigator = new G4Navigator();
    solidWorld = new G4Box("solidWorld", zWorld/2*cm, rWorld*cm, rWorld*cm); // define volume boundaries (length inputs should be half the actual length)
    logicWorld = new G4LogicalVolume(solidWorld, worldMat, "logicWorld"); // define volume material
    physWorld = new G4PVPlacement(0, G4ThreeVector(0., 0., 0.), logicWorld, "physWorld", 0, false, 0, checkDetectorsOverlaps);
    // parameters: rotation, volume center, logicalVolume, name, mother volume, boolean operations, copy number (ID), checkOverlaps
    navigator->SetWorldVolume(physWorld);

    // userLimits = new G4UserLimits();
    // userLimits->SetMaxAllowedStep(fStepLimit);

    if (sampleID != 0)
    {
        if (sampleID == 1 || sampleID == 2)
        {
            ConstructSample();
        } else if (sampleID == 3)
        {
            ConstructCylindricalPhantom();
        }
    }
    ConstructScintillators();
    ConstructSensitiveDetector();

    return physWorld;
}

void NSDetectorConstruction::ConstructSDandField() {}
