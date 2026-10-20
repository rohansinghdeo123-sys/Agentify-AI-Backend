import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session
from database import SessionLocal
from models import ContentChapter, ContentPage
from Logic.content_pipeline import import_concepts_for_chapter, approve_chapter, publish_chapter, chapter_report

def seed():
    db: Session = SessionLocal()
    try:
        # 1. Clean up existing if any
        slug = "ncert_class_11_chemistry_chapter_1_some_basic_concepts_of_chemistry"
        existing = db.query(ContentChapter).filter_by(slug=slug).first()
        if existing:
            db.delete(existing)
            db.commit()

        # 2. Setup Chapter
        chapter = ContentChapter(
            board="NCERT", class_level="11", subject="Chemistry",
            chapter_number=1, chapter_name="Some Basic Concepts of Chemistry",
            slug=slug, status="draft"
        )
        db.add(chapter)
        db.commit()
        db.refresh(chapter)
        
        # 3. Add Dummy Pages
        for i in range(1, 8):
            db.add(ContentPage(
                chapter_id=chapter.id, 
                page_number=i, 
                text=f"Master notes page {i}", 
                char_count=100, 
                extraction_quality=1.0,
                metadata_json={}
            ))
        db.commit()

        # 4. Prepare FULL concepts (No lines skipped)
        concepts = [
            {
                "concept_id": "atoms_and_molecules",
                "title": "Atoms and Molecules",
                "definition": "An atom is the smallest unit of an element that takes part in a chemical reaction.",
                "core_explanation": "Atoms contain protons (positive charge), neutrons (no charge) and electrons (negative charge). Protons and neutrons are in the nucleus; electrons occupy the region around it. A molecule is a group of two or more atoms chemically bonded together that can exist as a separate unit. H_2 contains two hydrogen atoms; O_2 contains two oxygen atoms; H_2O contains two hydrogen atoms and one oxygen atom; CO_2 contains one carbon atom and two oxygen atoms.",
                "key_points": ["Solid sodium chloride (NaCl) forms a repeating arrangement of ions.", "NaCl shows the simplest ratio of ions, described as a formula unit."],
                "formulas": ["H_2", "O_2", "H_2O", "CO_2"],
                "source_pages": [1], "difficulty_level": 2, "importance_level": "high", "typical_exam_weightage": "high"
            },
            {
                "concept_id": "nature_and_classification_of_matter",
                "title": "Nature and Classification of Matter",
                "definition": "Matter is anything that has mass and occupies space.",
                "core_explanation": "Matter is made of tiny particles such as atoms, molecules or ions. Solid: particles are close and mainly vibrate around fixed positions. Liquid: particles remain close but move past each other. Gas: particles are far apart and move rapidly.",
                "key_points": ["Melting: solid to liquid.", "Vaporisation: liquid to gas.", "A pure substance has a fixed chemical composition.", "A mixture contains substances physically mixed in variable proportions."],
                "source_pages": [1], "difficulty_level": 2, "importance_level": "high", "typical_exam_weightage": "high"
            },
            {
                "concept_id": "properties_of_matter_and_measurement",
                "title": "Properties of Matter and Measurement",
                "definition": "Physical properties can be observed without changing chemical identity. Chemical properties describe how a substance reacts.",
                "core_explanation": "A measurement includes a numerical value and a unit. For example, 5 cm means five centimetres. Units make measurements clear and comparable.",
                "source_pages": [2], "difficulty_level": 1, "importance_level": "medium", "typical_exam_weightage": "medium"
            },
            {
                "concept_id": "si_units_and_prefixes",
                "title": "SI Base Units, Derived Units and Prefixes",
                "definition": "The SI system provides standard units for measurement.",
                "core_explanation": "Seven SI base units: Length (m), mass (kg), time (s), electric current (A), temperature (K), amount of substance (mol), luminous intensity (cd). Derived units combine base units (e.g., area uses m^2, volume uses m^3, speed uses m s^{-1}, density uses kg m^{-3}).",
                "key_points": ["kilo (k) = 10^3", "centi (c) = 10^{-2}", "milli (m) = 10^{-3}", "micro (u) = 10^{-6}", "nano (n) = 10^{-9}"],
                "source_pages": [2], "difficulty_level": 2, "importance_level": "high", "typical_exam_weightage": "high"
            },
            {
                "concept_id": "mass_weight_volume_density",
                "title": "Mass, Weight, Volume, Density and Temperature",
                "definition": "Mass measures the amount of matter. Weight is gravitational force.",
                "core_explanation": "Mass (kg) does not change. Weight (N) = mass x gravitational acceleration. Volume is space occupied (SI unit m^3). 1 L = 1000 mL, 1 mL = 1 cm^3. Density = mass / volume.",
                "examples": ["If mass = 60 g and volume = 20 cm^3, density = 60 / 20 = 3 g cm^{-3}."],
                "formulas": ["W = m \\times g", "density = mass / volume"],
                "source_pages": [2], "difficulty_level": 2, "importance_level": "high", "typical_exam_weightage": "high"
            },
            {
                "concept_id": "uncertainty_and_scientific_notation",
                "title": "Uncertainty and Scientific Notation",
                "definition": "No measurement is perfectly exact due to instrument precision limits.",
                "core_explanation": "Scientific notation writes a number as a x 10^n, where a is at least 1 but less than 10. Multiply coefficients and add exponents. Divide coefficients and subtract exponents. For addition/subtraction, make powers of 10 the same first.",
                "examples": ["450000 = 4.5 x 10^5", "0.00072 = 7.2 x 10^{-4}", "(2 x 10^3) x (3 x 10^4) = 6 x 10^7"],
                "formulas": ["a \\times 10^n"],
                "source_pages": [3], "difficulty_level": 3, "importance_level": "high", "typical_exam_weightage": "high"
            },
            {
                "concept_id": "significant_figures",
                "title": "Significant Figures",
                "definition": "Significant figures indicate the precision of a measurement.",
                "core_explanation": "All non-zero digits count. Zeros between non-zero digits count. Leading zeros do not count. Trailing zeros after a decimal count. Trailing zeros in whole numbers can be ambiguous (use scientific notation). For multiplication/division, round to the fewest sig figs. For addition/subtraction, round to the least precise decimal place.",
                "examples": ["347 has 3 sig figs", "0.0042 has 2 sig figs", "2.500 has 4 sig figs", "1.500 x 10^3 has 4 sig figs"],
                "source_pages": [3], "difficulty_level": 3, "importance_level": "high", "typical_exam_weightage": "very_high"
            },
            {
                "concept_id": "dimensional_analysis",
                "title": "Dimensional Analysis",
                "definition": "Unit conversion uses a conversion fraction equal to 1 so unwanted units cancel.",
                "core_explanation": "Example: 2.5 km x (1000 m / 1 km) = 2500 m. Example: 500 mL x (1 L / 1000 mL) = 0.5 L. Check that the final unit matches the requested quantity.",
                "source_pages": [3], "difficulty_level": 2, "importance_level": "medium", "typical_exam_weightage": "medium"
            },
            {
                "concept_id": "laws_of_chemical_combinations",
                "title": "Laws of Chemical Combinations",
                "definition": "Fundamental laws governing chemical reactions.",
                "core_explanation": "Conservation of mass: mass is neither created nor destroyed. Definite proportions: a pure compound always contains the same elements in the same fixed proportion by mass. Multiple proportions: when two elements form more than one compound, masses combine in simple whole-number ratios. Gay-Lussac's law: reacting gases have simple whole-number volume ratios. Avogadro's law: equal volumes of gases at same T and P contain equal numbers of molecules.",
                "source_pages": [3, 4], "difficulty_level": 3, "importance_level": "very_high", "typical_exam_weightage": "very_high"
            },
            {
                "concept_id": "daltons_atomic_theory",
                "title": "Dalton's Atomic Theory",
                "definition": "Matter is made of atoms; atoms of different elements have different properties.",
                "core_explanation": "Compounds form when atoms combine in simple whole-number ratios. Ordinary chemical reactions rearrange atoms rather than creating or destroying them. Modern correction: atoms of the same element can have different masses because of isotopes.",
                "source_pages": [4], "difficulty_level": 2, "importance_level": "high", "typical_exam_weightage": "high"
            },
            {
                "concept_id": "atomic_molecular_formula_masses",
                "title": "Atomic, Average Atomic, Molecular and Formula Masses",
                "definition": "Unified atomic mass unit (u) is exactly one-twelfth of the mass of one carbon-12 atom.",
                "core_explanation": "Approximate value of 1 u is 1.6605 x 10^{-27} kg. Atomic mass is mass of an atom in u. Average atomic mass is a weighted average of isotope masses. Molecular mass is the sum of atomic masses in a molecule (e.g., H_2O = 18 u, CO_2 = 44 u). Formula mass is the sum of atomic masses in a formula unit (e.g., NaCl = 58.5 u).",
                "formulas": ["H_2O = 18\\text{ u}", "CO_2 = 44\\text{ u}", "NaCl = 58.5\\text{ u}"],
                "source_pages": [4], "difficulty_level": 3, "importance_level": "very_high", "typical_exam_weightage": "very_high"
            },
            {
                "concept_id": "mole_concept_and_molar_mass",
                "title": "Mole Concept and Molar Mass",
                "definition": "A mole is a counting unit (like a dozen) used because atoms are too small to count individually.",
                "core_explanation": "One mole contains exactly 6.02214076 x 10^{23} entities (Avogadro constant, N_A). Molar mass is the mass of one mole in g mol^{-1}. Moles = mass / molar mass. Particles = moles x N_A.",
                "examples": ["Moles in 36 g water: Molar mass H_2O = 18 g mol^{-1}. Moles = 36 / 18 = 2 mol.", "Mass of 0.5 mol CO_2 = 0.5 x 44 = 22 g."],
                "formulas": ["n = \\frac{m}{M}", "N = n \\times N_A"],
                "source_pages": [5], "difficulty_level": 3, "importance_level": "very_high", "typical_exam_weightage": "very_high"
            },
            {
                "concept_id": "percentage_composition",
                "title": "Percentage Composition",
                "definition": "Percentage by mass of an element in a compound.",
                "core_explanation": "Formula: (mass of element in one mole / molar mass of compound) x 100. Example for water (molar mass 18): Hydrogen percentage = (2 / 18) x 100 = 11.11%. Oxygen percentage = (16 / 18) x 100 = 88.89%.",
                "formulas": ["\\% \\text{ element} = \\frac{\\text{mass of element}}{\\text{molar mass}} \\times 100"],
                "source_pages": [5], "difficulty_level": 2, "importance_level": "high", "typical_exam_weightage": "high"
            },
            {
                "concept_id": "empirical_and_molecular_formulae",
                "title": "Empirical and Molecular Formulae",
                "definition": "Empirical formula gives the simplest whole-number ratio. Molecular formula gives actual atom counts.",
                "core_explanation": "Example: C_6H_{12}O_6 simplifies to CH_2O. To find empirical formula: assume 100g if given %, convert grams to moles, divide by smallest, convert to whole numbers.",
                "examples": ["40% C, 6.67% H, 53.33% O -> 3.33 mol C, 6.67 mol H, 3.33 mol O -> ratio 1:2:1 -> CH_2O."],
                "formulas": ["\\text{multiplier} = \\frac{\\text{molecular molar mass}}{\\text{empirical formula mass}}"],
                "source_pages": [5], "difficulty_level": 4, "importance_level": "very_high", "typical_exam_weightage": "very_high"
            },
            {
                "concept_id": "stoichiometry_and_limiting_reagent",
                "title": "Stoichiometry and Limiting Reagent",
                "definition": "Stoichiometry calculates amounts of reactants and products using a balanced equation.",
                "core_explanation": "A balanced equation has equal numbers of each element's atoms on both sides. Coefficients give the mole ratio. The limiting reagent is used up first and limits the amount of product; the excess reagent remains.",
                "examples": ["4 g H_2 = 2 mol. With excess O_2, 2 mol H_2O forms. Mass = 2 x 18 = 36 g.", "2H_2 + O_2 -> 2H_2O. 4 mol H_2 and 1 mol O_2 make 2 mol H_2O; O_2 is limiting."],
                "source_pages": [6], "difficulty_level": 4, "importance_level": "very_high", "typical_exam_weightage": "very_high"
            },
            {
                "concept_id": "concentration_of_solutions",
                "title": "Concentration of Solutions",
                "definition": "Concentration expresses how much solute is dissolved in a solvent.",
                "core_explanation": "Mass % = (mass of solute / mass of solution) x 100. Mole fraction (X_A) = moles of A / total moles. Molarity (M) = moles of solute / volume of solution in litres (mol L^{-1}). Molality (m) = moles of solute / mass of solvent in kg (mol kg^{-1}).",
                "key_points": ["Molarity uses solution volume and can change with temperature.", "Molality uses solvent mass and does not change with temperature."],
                "formulas": ["M = \\frac{n}{V(\\text{L})}", "m = \\frac{n}{\\text{kg solvent}}"],
                "source_pages": [6, 7], "difficulty_level": 4, "importance_level": "very_high", "typical_exam_weightage": "very_high"
            },
            {
                "concept_id": "quick_formula_sheet",
                "title": "Quick Formula Sheet",
                "definition": "Summary of core formulas for Chapter 1.",
                "core_explanation": "Density = mass / volume. Weight = mass x g. Moles = mass / molar mass. Particles = moles x N_A. Mass % = (mass of solute / mass of solution) x 100. Mole fraction = moles of component / total moles. Molarity = moles of solute / litres of solution. Molality = moles of solute / kg of solvent.",
                "source_pages": [7], "difficulty_level": 1, "importance_level": "high", "typical_exam_weightage": "high"
            }
        ]

        # 5. Import the concepts
        import_concepts_for_chapter(db, chapter.id, concepts, replace=True)
        
        # 6. Approve and Publish
        approve_chapter(db, chapter.id, approved_by="Amit")
        publish_chapter(db, chapter.id, published_by="Amit")
        
        # 7. Commit the final state to the Neon Database
        db.commit()
        
        print("✅ Chapter 1 successfully seeded, approved, and published to the DATABASE!")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Approval failed: {e}")
        try:
            report = chapter_report(db, chapter.id)
            print("\n--- VALIDATION REPORT ---")
            print(json.dumps(report["report"], indent=2))
        except:
            pass
    finally:
        db.close()

if __name__ == "__main__":
    seed()