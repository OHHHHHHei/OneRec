# R650a Error Item Diagnosis

- examples: 4533
- v2 hit@10: 663/4533 = 0.1463
- R650a hit@10: 600/4533 = 0.1324
- top10 losses: 206, gains: 143, net: -63

## Loss Rank Destination
- 11-20: 73 (35.4%)
- 21-50: 57 (27.7%)
- >50: 76 (36.9%)

## Loss Target Families
- 3d_filament: 90 (43.7%)
- other: 39 (18.9%)
- gauge_meter: 25 (12.1%)
- connector_fitting: 15 (7.3%)
- tape: 14 (6.8%)
- adhesive_epoxy: 8 (3.9%)
- ventilation_fan: 5 (2.4%)
- metadata_placeholder: 4 (1.9%)
- test_strip: 3 (1.5%)
- fastener: 3 (1.5%)

## Gain Target Families
- 3d_filament: 66 (46.2%)
- other: 32 (22.4%)
- gauge_meter: 14 (9.8%)
- adhesive_epoxy: 13 (9.1%)
- tape: 11 (7.7%)
- connector_fitting: 5 (3.5%)
- ventilation_fan: 1 (0.7%)
- fastener: 1 (0.7%)

## Pred1 Families On Losses
- 3d_filament: 85 (41.3%)
- other: 69 (33.5%)
- gauge_meter: 14 (6.8%)
- connector_fitting: 11 (5.3%)
- adhesive_epoxy: 11 (5.3%)
- fastener: 7 (3.4%)
- tape: 5 (2.4%)
- ventilation_fan: 2 (1.0%)
- metadata_placeholder: 2 (1.0%)

## Worst Items
- item 273 | n=3 | v2=1.000 -> r650a=0.000 | delta=-1.000 | Gorilla Original Gorilla Glue, Waterproof Polyurethane Glue, 2 ounce Bottle, Brown, (Pack of 4) | pred1 mostly: Gorilla 5000408  Original Gorilla Glue, Waterproof Polyurethane Glue, 4 ounce Bottle, Bro…
- item 3475 | n=11 | v2=0.818 -> r650a=0.091 | delta=-0.727 | 3D Solutech Real Orange 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, … | pred1 mostly: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0.03 mm, 2.…
- item 98 | n=5 | v2=0.600 -> r650a=0.000 | delta=-0.600 | Industrial & Scientific" /> | pred1 mostly: Loctite Liquid Professional Super Glue  20-Gram Bottle (1365882)
- item 2909 | n=31 | v2=0.581 -> r650a=0.065 | delta=-0.516 | 3D Solutech Silver Metal 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm,… | pred1 mostly: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0.03 mm, 2.…
- item 3522 | n=14 | v2=0.643 -> r650a=0.143 | delta=-0.500 | 3D Solutech Real Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.… | pred1 mostly: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0.03 mm, 2.…
- item 560 | n=6 | v2=0.667 -> r650a=0.167 | delta=-0.500 | Litmus pH Test Strips, Universal Application (pH 1-14), 2 Packs of 100 Strips | pred1 mostly: Phresh Duct Silencer 8 in x 24 in
- item 450 | n=4 | v2=0.500 -> r650a=0.000 | delta=-0.500 | X-Treme Tape TPE-X36ZLB Silicone Rubber Self Fusing Tape, 1" x 36', Triangular, Black | pred1 mostly: American Terminal E-FFR250N-100 22/18-Gauge Economy Nylon Fully-Insulated Female Quick Di…
- item 3442 | n=9 | v2=0.556 -> r650a=0.111 | delta=-0.444 | 3D Solutech Real Purple 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, … | pred1 mostly: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0.03 mm, 2.…
- item 201 | n=8 | v2=0.500 -> r650a=0.125 | delta=-0.375 | ColorConnex Coupler & Plug Kit (7 Piece), Industrial Type D, 1/4 in. NPT, Red - A73457D | pred1 mostly: Yueton 100pcs Female Fully Insulated Wire Crimp Terminal Nylon Quick Connectors Wiring Sp…
- item 2807 | n=8 | v2=0.875 -> r650a=0.500 | delta=-0.375 | VIVOSUN 6 Inch 440 CFM Inline Duct Ventilation Fan with Variable Speed Controller | pred1 mostly: AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital Hygrometer and Humidity G…
- item 2703 | n=14 | v2=0.429 -> r650a=0.071 | delta=-0.357 | eSUN 3D 1.75mm PETG Black Filament 1kg (2.2lb), PETG 3D Printer Filament, 1.75mm Solid Opaque Black | pred1 mostly: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, …
- item 3016 | n=17 | v2=0.529 -> r650a=0.176 | delta=-0.353 | HATCHBOX 3D Printer Filament, Dimensional Accuracy +/- 0.03mm, 1.75 mm, 1 kg Spool, Wood | pred1 mostly: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, …
- item 3210 | n=6 | v2=0.667 -> r650a=0.333 | delta=-0.333 | Ancor Heat Shrink Ring Terminals | pred1 mostly: Neiko 10194A Titanium Step Drill Bit, High Speed Steel | 1/4" to 1-3/8" | Total 10 Step S…
- item 2871 | n=3 | v2=0.333 -> r650a=0.000 | delta=-0.333 | 3M 468MP Adhesive Transfer Tape, 4" width x 5yd length (1 roll) | pred1 mostly: eSUN 1.75mm White ABS+ 3D Printer filament 1kg Spool (2.2lbs), White
- item 45 | n=3 | v2=0.333 -> r650a=0.000 | delta=-0.333 | Permatex 80050 Clear RTV Silicone Adhesive Sealant, 3 oz | pred1 mostly: ELEGOO 5 Sets 28BYJ-48 ULN2003 5V Stepper Motor + ULN2003 Driver Board for Arduino
- item 1376 | n=3 | v2=0.333 -> r650a=0.000 | delta=-0.333 | Century Drill and Tool 97205 Plug Hand Pipe Tap, 3/4-14 NPT | pred1 mostly: Industrial & Scientific" />
- item 2793 | n=3 | v2=0.667 -> r650a=0.333 | delta=-0.333 | First Aid Only Splinter Out, 10 Per Box | pred1 mostly: 3M Vetbond Tissue Adhesive, 3ml Bottles w/MSDS
- item 734 | n=3 | v2=0.333 -> r650a=0.000 | delta=-0.333 | microtivity IL188 5mm Assorted Clear LED w/Resistors (8 Colors, Pack of 80) | pred1 mostly: Ribbed Plastic Drywall Anchor Kit with Screws and Masonry Drill Bit, #10-12 x 1
- item 1042 | n=3 | v2=0.333 -> r650a=0.000 | delta=-0.333 | uxcell Mandrel Mounted White Conical Felt Point Polishing Tool | pred1 mostly: American Terminal E-FMB250N-100 16/14-Gauge Economy Nylon Fully-Insulated Male Quick Disc…
- item 3333 | n=3 | v2=0.333 -> r650a=0.000 | delta=-0.333 | E-Z Lok Threaded Insert, Brass, Knife Thread, 3/8"-16 Internal Threads, 0.625" Length (Pack of 10) | pred1 mostly: Bostitch BTFP72326 Regulator and Gauge Kit with 1/4-Inch NPT Thread

## Best Items
- item 3466 | n=11 | v2=0.091 -> r650a=0.909 | delta=0.818 | 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0… | pred1 mostly: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0.03 mm, 2.…
- item 326 | n=3 | v2=0.000 -> r650a=0.667 | delta=0.667 | J-B Weld 8276 KwikWeld Quick Setting Steel Reinforced Epoxy - 2 oz. | pred1 mostly: Marrywindix 6pcs ESD Precision Anti-Static Tweezers, Tweezers Stainless Steel Tweezers wi…
- item 2311 | n=3 | v2=0.000 -> r650a=0.667 | delta=0.667 | TEMCo 3/4" Marine Heat Shrink Tube 3:1 Adhesive Glue Lined 4 ft RED | pred1 mostly: TEMCo 1/2" Marine Heat Shrink Tube 3:1 Adhesive Glue Lined 4 ft RED
- item 3546 | n=3 | v2=0.000 -> r650a=0.667 | delta=0.667 | Inland 1.75mm Orange PLA 3D Printer Filament - 1kg Spool (2.2 lbs) | pred1 mostly: Inland 1.75mm Gray PLA 3D Printer Filament - 1kg Spool (2.2 lbs)
- item 125 | n=5 | v2=0.000 -> r650a=0.600 | delta=0.600 | Gorilla 6100101 Tape Handy Roll, 1-Pack, Black | pred1 mostly: Monoprice Hook & Loop Fastening Tape 5 yard/roll, 0.75-inch - Blue (105830)
- item 3453 | n=12 | v2=0.000 -> r650a=0.500 | delta=0.500 | 3D Solutech Real Gold 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.… | pred1 mostly: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0.03 mm, 2.…
- item 970 | n=5 | v2=0.000 -> r650a=0.400 | delta=0.400 | Wixey WR25 Mini Digital Height Gauge | pred1 mostly: Red Devil 0325 Steel Wool, 2 Medium Course, 8 Pads
- item 675 | n=5 | v2=0.200 -> r650a=0.600 | delta=0.400 | J-B Weld 8265S Original Cold-Weld Steel Reinforced Epoxy - 2 oz. | pred1 mostly: Install Bay Copper Ring Terminal Connector 8 Gauge 1/4 Inch 25 Pack - CUR814
- item 1035 | n=3 | v2=0.000 -> r650a=0.333 | delta=0.333 | Anderson Metals Brass Pipe Fitting, 90 Degree Barstock Street Elbow, 3/8" Male Pipe x 3/8" Female P… | pred1 mostly: SE 82331TF 30-Piece Set of Titanium-Coated Diamond Burrs, Grits 120-150
- item 1285 | n=3 | v2=0.000 -> r650a=0.333 | delta=0.333 | Anderson Metals Brass Pipe Fitting, Coupling, 1/2" x 1/2" Female Pipe | pred1 mostly: Anderson Metals 57001-0402 Brass Hose Fitting, Adapter, 1/4" Barb x 1/8" NPT Male Pipe
- item 467 | n=3 | v2=0.000 -> r650a=0.333 | delta=0.333 | 3M Utility Duct Tape 2929 Silver, 1.88 in x 50 yd 5.8 mils (Pack of 1) | pred1 mostly: The Hillman Group 591519 Small Small Wood Screw Assortment, 195-Pack
- item 333 | n=3 | v2=0.000 -> r650a=0.333 | delta=0.333 | DEWALT DW4930 4-Inch by 5/8-Inch-11 Full Cable Twist Wire Wheel/Carbon Steel .020-Inch | pred1 mostly: Inkbird ITC-1000 Dual Stage Digital Temperature Controller Fahrenheit  Celsius Thermostat…
- item 353 | n=3 | v2=0.000 -> r650a=0.333 | delta=0.333 | Scotch Aluminum Foil Tape 3311 Silver, 2 in x 10 yd 3.6 mil (Pack of 1) | pred1 mostly: Deflecto Early Warning Road Safety Triangle Kit, Reflective, 3-Pack (73-0711-00)
- item 51 | n=7 | v2=0.000 -> r650a=0.286 | delta=0.286 | HATCHBOX 1 Spool 3D Printer Filament Tabletop Wall Mount Rack | pred1 mostly: HATCHBOX 3D Printer Filament, Dimensional Accuracy +/- 0.03mm, 1.75 mm, 1 kg Spool, Wood
- item 3112 | n=32 | v2=0.000 -> r650a=0.250 | delta=0.250 | 3D Solutech Real Black 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2… | pred1 mostly: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0.03 mm, 2.…
- item 57 | n=16 | v2=0.000 -> r650a=0.250 | delta=0.250 | Gorilla 2 Part Epoxy, 5 Minute Set, .85 ounce Syringe, Clear | pred1 mostly: AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital Hygrometer and Humidity G…
- item 1554 | n=8 | v2=0.000 -> r650a=0.250 | delta=0.250 | Inland 1.75mm Peak Green PLA 3D Printer Filament - 1kg Spool (2.2 lbs) | pred1 mostly: Inland 1.75mm Black PLA 3D Printer Filament - 1kg Spool (2.2 lbs)
- item 1553 | n=8 | v2=0.250 -> r650a=0.500 | delta=0.250 | Inland 1.75mm Red PLA 3D Printer Filament - 1kg Spool (2.2 lbs) | pred1 mostly: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, …
- item 1098 | n=4 | v2=0.000 -> r650a=0.250 | delta=0.250 | J-B Weld 50112 ClearWeld Quick-Setting Epoxy Syringe - Dries Clear - 25 ml | pred1 mostly: AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital Hygrometer and Humidity G…
- item 2106 | n=4 | v2=0.000 -> r650a=0.250 | delta=0.250 | Noga NG8150 Heavy Duty Deburr Tool, with 10 S10 blades | pred1 mostly: 2 Meters PTFE Teflon Bowden Tube 1.75 Filament 3D printer RepRap Rostock Kossel

## Top Replacement Pairs On Losses
- n=7 | target 2909: 3D Solutech Silver Metal 3D Printer PLA Filament 1.75MM Filament, Dimensional A… -> pred1 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0… | pred1_lcp=1
- n=6 | target 3522: 3D Solutech Real Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accu… -> pred1 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0… | pred1_lcp=1
- n=4 | target 2909: 3D Solutech Silver Metal 3D Printer PLA Filament 1.75MM Filament, Dimensional A… -> pred1 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Ac… | pred1_lcp=1
- n=4 | target 3475: 3D Solutech Real Orange 3D Printer PLA Filament 1.75MM Filament, Dimensional Ac… -> pred1 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0… | pred1_lcp=2
- n=4 | target 3442: 3D Solutech Real Purple 3D Printer PLA Filament 1.75MM Filament, Dimensional Ac… -> pred1 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0… | pred1_lcp=2
- n=3 | target 2807: VIVOSUN 6 Inch 440 CFM Inline Duct Ventilation Fan with Variable Speed Controll… -> pred1 176: URBEST 530 Pcs 2:1 Heat Shrink Tubing Tube Sleeving Wrap Cable Wire 5 Color 8 S… | pred1_lcp=0
- n=3 | target 3016: HATCHBOX 3D Printer Filament, Dimensional Accuracy +/- 0.03mm, 1.75 mm, 1 kg Sp… -> pred1 2507: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool,… | pred1_lcp=1
- n=2 | target 1847: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool,… -> pred1 50: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool,… | pred1_lcp=0
- n=2 | target 2909: 3D Solutech Silver Metal 3D Printer PLA Filament 1.75MM Filament, Dimensional A… -> pred1 2697: 3D Solutech Real White 3D Printer PLA Filament 1.75MM Filament, Dimensional Acc… | pred1_lcp=1
- n=2 | target 2507: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool,… -> pred1 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0… | pred1_lcp=0
- n=2 | target 3016: HATCHBOX 3D Printer Filament, Dimensional Accuracy +/- 0.03mm, 1.75 mm, 1 kg Sp… -> pred1 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0… | pred1_lcp=0
- n=2 | target 3522: 3D Solutech Real Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accu… -> pred1 2697: 3D Solutech Real White 3D Printer PLA Filament 1.75MM Filament, Dimensional Acc… | pred1_lcp=1
- n=2 | target 182: AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital Hygrometer and … -> pred1 119: Rubbermaid Commercial BRUTE Heavy-Duty Round Waste/Utility Container with Venti… | pred1_lcp=0
- n=1 | target 450: X-Treme Tape TPE-X36ZLB Silicone Rubber Self Fusing Tape, 1" x 36', Triangular,… -> pred1 58: American Terminal E-FFR250N-100 22/18-Gauge Economy Nylon Fully-Insulated Femal… | pred1_lcp=0
- n=1 | target 182: AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital Hygrometer and … -> pred1 151: Monoprice Hook & Loop Fastening Tape 5 yard/roll, 0.75-inch - Blue (105830) | pred1_lcp=0
- n=1 | target 3475: 3D Solutech Real Orange 3D Printer PLA Filament 1.75MM Filament, Dimensional Ac… -> pred1 2697: 3D Solutech Real White 3D Printer PLA Filament 1.75MM Filament, Dimensional Acc… | pred1_lcp=2
- n=1 | target 182: AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital Hygrometer and … -> pred1 764: Shop-Vac 9050300 2.5-Inch by 8-Foot Hose | pred1_lcp=0
- n=1 | target 182: AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital Hygrometer and … -> pred1 1644: O-Cedar Heavy Duty Scrunge Scrub Sponge (Pack of 6) | pred1_lcp=0
- n=1 | target 182: AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital Hygrometer and … -> pred1 1286: Accessbuy 180-Piece Rubber Grommet Shop Assortment | pred1_lcp=0
- n=1 | target 465: Dorman Hardware 4-9845 Peggable Wire Basket Set, 3-Pack -> pred1 338: Etekcity Lasergrip 1080 Non-Contact Digital Laser Infrared Thermometer Temperat… | pred1_lcp=0
- n=1 | target 1911: uxcell 1 Meter 65 Flat width 40mm Dia Ratio 2:1 Heat Shrinkable Shrinking Tube … -> pred1 40: Forney 20859 Cutting Fluid, Industrial Pro Tap Magic, 1-Gallon | pred1_lcp=0
- n=1 | target 2419: HATCHBOX 3D ABS-1KG1.75-BLU ABS 3D Printer Filament, Dimensional Accuracy +/- 0… -> pred1 2157: eSUN 3D 1.75mm PETG Blue Filament 1kg (2.2lb), PETG 3D Printer Filament, 1.75mm… | pred1_lcp=1
- n=1 | target 560: Litmus pH Test Strips, Universal Application (pH 1-14), 2 Packs of 100 Strips -> pred1 218: Phresh Duct Silencer 8 in x 24 in | pred1_lcp=0
- n=1 | target 2912: Inland 1.75mm Black PLA 3D Printer Filament - 1kg Spool (2.2 lbs) -> pred1 2158: eSUN 3D 1.75mm PETG Green Filament 1kg (2.2lb), PETG 3D Printer Filament, Semi-… | pred1_lcp=0
- n=1 | target 201: ColorConnex Coupler & Plug Kit (7 Piece), Industrial Type D, 1/4 in. NPT, Red -… -> pred1 157: Yueton 100pcs Female Fully Insulated Wire Crimp Terminal Nylon Quick Connectors… | pred1_lcp=0
- n=1 | target 157: Yueton 100pcs Female Fully Insulated Wire Crimp Terminal Nylon Quick Connectors… -> pred1 457: CRC 03400 16oz Truetap Cutting Fluid Heavy Duty Bottle | pred1_lcp=0
- n=1 | target 175: Gorilla Crystal Clear Duct Tape, 1.88&rdquo; x 9 yd, Clear, (Pack of 1) -> pred1 340: Break-Free CLP-4 Cleaner Lubricant Preservative Squeeze Bottle (4 -Fluid Ounce) | pred1_lcp=0
- n=1 | target 217: Ancor Tinned Copper Lugs 8 AWG - 4/0 AWG -> pred1 997: The Hillman Group 4839 1/8 Wire Rope Clip, 10-Pack | pred1_lcp=0
- n=1 | target 3210: Ancor Heat Shrink Ring Terminals -> pred1 178: Neiko 10194A Titanium Step Drill Bit, High Speed Steel | 1/4" to 1-3/8" | Total… | pred1_lcp=0
- n=1 | target 182: AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital Hygrometer and … -> pred1 521: Gardner Bender 19-001 WireGard Screw-On Wire Connectors, 22-16 AWG, Electrical … | pred1_lcp=0