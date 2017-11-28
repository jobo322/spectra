/**
 * Created by acastillo on 5/7/16.
 */

const FS = require('fs');
const OCLE = require('openchemlib-extended-minimal');
const autoassigner = require('../src/index');
const predictor = require('../../nmr-predictor/src/index');
require('should');


function loadFile(filename) {
    return FS.readFileSync(__dirname + filename).toString();
}

describe('ssss.', function () {
    var peakPicking = [{from: 7.256290906230099, to: 7.318706060368815, integral: 1.968363767180158, signal: [{nbAtoms: 0, diaID: [], multiplicity: 'm', peak: [{x: 7.260054634117861, intensity: 3503694.35930736, width: 0.0012545759625872677}, {x: 7.264132005996269, intensity: 19862399.272727273, width: 0.0012545759625872677}, {x: 7.2672684459027375, intensity: 65619616.63636364, width: 0.0009409319719404508}, {x: 7.269150309846618, intensity: 93856276.85281387, width: 0.0018818639438809015}, {x: 7.273541325715674, intensity: 39090631.86580087, width: 0.0018818639438809015}, {x: 7.284518865388312, intensity: 59981487.84415585, width: 0.0012545759625872677}, {x: 7.287028017313487, intensity: 98047775.96103898, width: 0.0015682199532340846}, {x: 7.288909881257368, intensity: 94820341.74458875, width: 0.0015682199532340846}, {x: 7.291419033182542, intensity: 45483286.61904763, width: 0.0006272879812936338}, {x: 7.293928185107717, intensity: 21708886.060606062, width: 0.0009409319719404508}, {x: 7.3014556408832405, intensity: 21506116.320346322, width: 0.0015682199532340846}, {x: 7.305533012761649, intensity: 47441063.22077922, width: 0.0018818639438809015}, {x: 7.308042164686824, intensity: 34667062.90909091, width: 0.0012545759625872677}, {x: 7.309924028630705, intensity: 18594008.722943723, width: 0.0003136439906468169}, {x: 7.314942332481054, intensity: 4761865.341991342, width: 0.0012545759625872677}], kind: '', remark: '', delta: 7.282714716436958}], signalID: '1H_1', _highlight: ['1H_1']}, {from: 7.150906525372768, to: 7.223044643221536, integral: 2.765675360672561, signal: [{nbAtoms: 0, diaID: [], multiplicity: 'm', peak: [{x: 7.155611185232471, intensity: 9719681.800865801, width: 0.0015682199532340846}, {x: 7.159374913120232, intensity: 19998906.614718616, width: 0.0018818639438809015}, {x: 7.163138641007994, intensity: 13300704.151515154, width: 0.0018818639438809015}, {x: 7.171920672746105, intensity: 13599470.532467531, width: 0.0015682199532340846}, {x: 7.177566264577748, intensity: 51568725.103896104, width: 0.003450083897114986}, {x: 7.182584568428097, intensity: 23718255.406926405, width: 0.0015682199532340846}, {x: 7.187289228287799, intensity: 12514421.186147187, width: 0.0009409319719404508}, {x: 7.190739312184914, intensity: 36029976.48917749, width: 0.0006272879812936338}, {x: 7.192307532138148, intensity: 60719994.038961045, width: 0.0012545759625872677}, {x: 7.194189396082029, intensity: 81360531.52380954, width: 0.0009409319719404508}, {x: 7.19607126002591, intensity: 92643358.4891775, width: 0.0009409319719404508}, {x: 7.199207699932378, intensity: 70888134.55411257, width: 0.003450083897114986}, {x: 7.205794223735961, intensity: 20645331.203463204, width: 0.0006272879812936338}, {x: 7.207362443689195, intensity: 29206344.65800866, width: 0.0009409319719404508}, {x: 7.213635323502132, intensity: 112580044.51515152, width: 0.003136439906468169}, {x: 7.217085407399247, intensity: 89493869.27705629, width: 0.0012545759625872677}, {x: 7.218653627352481, intensity: 68857800.87445888, width: 0.0009409319719404508}], kind: '', remark: '', delta: 7.201523074917693}], signalID: '1H_2', _highlight: ['1H_2']}, {from: 2.5650431139173557, to: 2.642826823597876, integral: 1.9766930175543398, signal: [{nbAtoms: 0, diaID: [], multiplicity: 'q', peak: [{x: 2.575154345824339, intensity: 39203735.865800865, width: 0.003763727887767132}, {x: 2.594600273244469, intensity: 106250482.94805196, width: 0.004077371878414393}, {x: 2.6134189126833047, intensity: 109431469.66233768, width: 0.004077371878414393}, {x: 2.6322375521221404, intensity: 40437659.02597403, width: 0.003450083897119871}], kind: '', remark: '', j: [{coupling: 7.65449122379232, multiplicity: 'q'}], delta: 2.6039349687576157}], signalID: '1H_3', _highlight: ['1H_3']}, {from: 1.16673039830042, to: 1.21440428487877, integral: 2.7920656081490676, signal: [{nbAtoms: 0, diaID: [], multiplicity: 't', peak: [{x: 1.1712838436877855, intensity: 176611403.03896105, width: 0.0015682199532351948}, {x: 1.190729771107902, intensity: 317420136.26839834, width: 0.0018818639438822338}, {x: 1.2095484105467242, intensity: 205807990.6839827, width: 0.0018818639438822338}], kind: '', remark: '', j: [{coupling: 7.654491223786863, multiplicity: 't'}], delta: 1.190567341589595}], signalID: '1H_4', _highlight: ['1H_4']}];
    var cosyZones = [{nucleusX: '1H', nucleusY: '1H', resolutionX: -0.008408698925431128, resolutionY: -0.0084169082455412, shiftX: 7.198285176388374, shiftY: 7.279053936671747, fromTo: [{from: 7.150433809758376, to: 7.209352239765698}, {from: 7.27663496124714, to: 7.2864816977973526}], peaks: [{x: 7.153023584863875, y: 7.2864816977973526, z: 407375316}, {x: 7.202003112113596, y: 7.2827179342941495, z: 1881943893}, {x: 7.209352239765698, y: 7.27663496124714, z: 77647382}, {x: 7.150433809758376, y: 7.285051869492682, z: 51096649}], _highlight: ['198_0'], signalID: '198_0'}, {nucleusX: '1H', nucleusY: '1H', resolutionX: -0.008408698925431128, resolutionY: -0.0084169082455412, shiftX: 7.279053936671747, shiftY: 7.279053936671747, fromTo: [{from: 7.276687588345495, to: 7.276931748453136}, {from: 7.27663496124714, to: 7.276879121055215}], peaks: [{x: 7.276931748453136, y: 7.276879121055215, z: 2743098615}, {x: 7.276687588345495, y: 7.27663496124714, z: 172103449}], _highlight: ['198_1'], signalID: '198_1'}, {nucleusX: '1H', nucleusY: '1H', resolutionX: -0.008408698925431128, resolutionY: -0.0084169082455412, shiftX: 7.279053936671747, shiftY: 7.198285176388374, fromTo: [{from: 7.276687588345495, to: 7.286534336976937}, {from: 7.150381337564022, to: 7.209299695282811}], peaks: [{x: 7.286534336976937, y: 7.152971109492059, z: 407375316}, {x: 7.2829095903692656, y: 7.201610246586484, z: 1906598506}, {x: 7.285104506917969, y: 7.150381337564022, z: 51096649}, {x: 7.276687588345495, y: 7.209299695282811, z: 77647382}], _highlight: ['198_2'], signalID: '198_2'}, {nucleusX: '1H', nucleusY: '1H', resolutionX: -0.008408698925431128, resolutionY: -0.0084169082455412, shiftX: 7.198285176388374, shiftY: 7.198285176388374, fromTo: [{from: 7.165418735271049, to: 7.209352239765698}, {from: 7.165366244691305, to: 7.209299695282811}], peaks: [{x: 7.165418735271049, y: 7.165366244691305, z: 383172242}, {x: 7.203768096767339, y: 7.203715559135779, z: 3102858110}, {x: 7.167267646903326, y: 7.167215154055104, z: 60057256}, {x: 7.209352239765698, y: 7.209299695282811, z: 243685217}], _highlight: ['198_3'], signalID: '198_3'}, {nucleusX: '1H', nucleusY: '1H', resolutionX: -0.008408698925431128, resolutionY: -0.0084169082455412, shiftX: 2.598996149574079, shiftY: 7.198285176388374, fromTo: [{from: 2.605297780622114, to: 2.605297780622114}, {from: 7.200882787037269, to: 7.200882787037269}], peaks: [{x: 2.605297780622114, y: 7.200882787037269, z: 4335105}, {x: 2.605297780622114, y: 7.200882787037269, z: 4335105}], _highlight: ['198_4'], signalID: '198_4'}, {nucleusX: '1H', nucleusY: '1H', resolutionX: -0.008408698925431128, resolutionY: -0.0084169082455412, shiftX: 3.421304349396864, shiftY: 3.421304349396864, fromTo: [{from: 3.4212489178427576, to: 3.4217388821521464}, {from: 3.421201021081032, to: 3.4216909847892705}], peaks: [{x: 3.4212489178427576, y: 3.421201021081032, z: 1711714603}, {x: 3.4217388821521464, y: 3.4216909847892705, z: 291976049}], _highlight: ['198_5'], signalID: '198_5'}, {nucleusX: '1H', nucleusY: '1H', resolutionX: -0.008408698925431128, resolutionY: -0.0084169082455412, shiftX: 1.1846035971267705, shiftY: 2.598996149574079, fromTo: [{from: 1.1644784007344517, to: 1.2080892975913367}, {from: 2.57158325198961, to: 2.6220847014628568}], peaks: [{x: 1.1644784007344517, y: 2.597541222517262, z: 2801230951}, {x: 1.203343489745686, y: 2.597513741058547, z: 3193792690}, {x: 1.1660047047289641, y: 2.57158325198961, z: 153964080}, {x: 1.2080892975913367, y: 2.57158325198961, z: 155848128}, {x: 1.1660047047289641, y: 2.6220847014628568, z: 144111512}, {x: 1.2080892975913367, y: 2.6220847014628568, z: 151243635}], _highlight: ['198_6'], signalID: '198_6'}, {nucleusX: '1H', nucleusY: '1H', resolutionX: -0.008408698925431128, resolutionY: -0.0084169082455412, shiftX: 7.198285176388374, shiftY: 2.598996149574079, fromTo: [{from: 7.200935321193224, to: 7.200935321193224}, {from: 2.605250884971775, to: 2.605250884971775}], peaks: [{x: 7.200935321193224, y: 2.605250884971775, z: 4335105}, {x: 7.200935321193224, y: 2.605250884971775, z: 4335105}], _highlight: ['198_7'], signalID: '198_7'}, {nucleusX: '1H', nucleusY: '1H', resolutionX: -0.008408698925431128, resolutionY: -0.0084169082455412, shiftX: 2.598996149574079, shiftY: 2.598996149574079, fromTo: [{from: 2.596581819799498, to: 2.5968808620496393}, {from: 2.5965349348429942, to: 2.5968339767262334}], peaks: [{x: 2.596581819799498, y: 2.5965349348429942, z: 3235218600}, {x: 2.5968808620496393, y: 2.5968339767262334, z: 100211648}], _highlight: ['198_8'], signalID: '198_8'}, {nucleusX: '1H', nucleusY: '1H', resolutionX: -0.008408698925431128, resolutionY: -0.0084169082455412, shiftX: 2.5068558720513483, shiftY: 2.5068558720513483, fromTo: [{from: 2.5042947577524197, to: 2.507185259953964}, {from: 2.5042479860252804, to: 2.507138484680394}], peaks: [{x: 2.507185259953964, y: 2.507138484680394, z: 92902618}, {x: 2.5042947577524197, y: 2.5042479860252804, z: 11313862}], _highlight: ['198_9'], signalID: '198_9'}, {nucleusX: '1H', nucleusY: '1H', resolutionX: -0.008408698925431128, resolutionY: -0.0084169082455412, shiftX: 2.598996149574079, shiftY: 1.1846035971267705, fromTo: [{from: 2.571630106332215, to: 2.622131617767063}, {from: 1.1645197262811164, to: 1.2080441162119353}], peaks: [{x: 2.597782250576457, y: 1.1645197262811164, z: 2785499075}, {x: 2.597560627215974, y: 1.2032983141890385, z: 3193792690}, {x: 2.571630106332215, y: 1.16595957498423, z: 153964080}, {x: 2.622131617767063, y: 1.16595957498423, z: 144111512}, {x: 2.571630106332215, y: 1.2080441162119353, z: 155848128}, {x: 2.622131617767063, y: 1.2080441162119353, z: 151243635}], _highlight: ['198_10'], signalID: '198_10'}, {nucleusX: '1H', nucleusY: '1H', resolutionX: -0.008408698925431128, resolutionY: -0.0084169082455412, shiftX: 1.1846035971267705, shiftY: 1.1846035971267705, fromTo: [{from: 1.1828385418739131, to: 1.1846048843868555}, {from: 1.182793391475312, to: 1.183628237189457}], peaks: [{x: 1.1846048843868555, y: 1.183628237189457, z: 9025793295}, {x: 1.1828385418739131, y: 1.182793391475312, z: 424898800}], _highlight: ['198_11'], signalID: '198_11'}];
    var molecule = OCLE.Molecule.fromSmiles('CCc1ccccc1');
    molecule.addImplicitHydrogens();
    var molfile = molecule.toMolfile();
    //var nH = molecule.getMolecularFormula().formula.replace(/.*H([0-9]+).*/, '$1') * 1;

    const db = JSON.parse(loadFile('/../../nmr-predictor/data/h1.json'));
    predictor.setDb(db, 'proton', 'proton');

    it('Known patterns for ethylbenzene', function () {
        var result = autoassigner({general: {molfile: molfile},
            spectra: {nmr: [{nucleus: 'H', experiment: '1d', range: peakPicking},
                {nucleus: ['H', 'H'], experiment: 'cosy', region: cosyZones}]}},
        {minScore: 0.8, maxSolutions: 3000, errorCS: 1, predictor: predictor, condensed: true, OCLE: OCLE}
        );
        result.getAssignments().length.should.equal(2);
        result.getAssignments()[0].score.should.greaterThan(0.8);
    });

    it('condensed for ethylbenzene', function () {
        var result = autoassigner({general: {molfile: molfile},
            spectra: {nmr: [{nucleus: 'H', experiment: '1d', range: peakPicking},
                {nucleus: ['H', 'H'], experiment: 'cosy', region: cosyZones}]}},
        {minScore: 0.9, maxSolutions: 3000, errorCS: 0, predictor: predictor, condensed: true, OCLE: OCLE}
        ).getAssignments();
        result.length.should.equal(4);
        result[0].score.should.greaterThan(0.9);
    });

    it('condensed for ethylbenzene from molfile', function () {
        var result = autoassigner({
            general: {molfile: molecule.toMolfileV3()},
            spectra: {
                nmr: [{
                    nucleus: 'H',
                    experiment: '1d',
                    range: peakPicking,
                    solvent: 'unknown'
                }]
            }
        },
        {
            minScore: 1,
            maxSolutions: 3000,
            errorCS: 0,
            predictor: predictor,
            condensed: true,
            OCLE: OCLE,
            levels: [5, 4, 3]
        }
        ).getAssignments();
        result.length.should.equal(12);
        result[0].score.should.equal(1);
        /*var result = autoassigner({general: {molfile: molfile},
            spectra: {nmr: [{nucleus: 'H', experiment: '1d', range: peakPicking}]}},
        {minScore: 1, maxSolutions: 3000, errorCS: 0, predictor: predictor, condensed: true, OCLE: OCLE}
        ).getAssignments();
        result.length.should.equal(12);
        result[0].score.should.equal(1);*/

    });
});
