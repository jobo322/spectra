'use strict';

const predictor = require('..');
const fs = require('fs');

const db1H = JSON.parse(fs.readFileSync(__dirname + '/../data/h1.json', 'utf8'));
const db13C = JSON.parse(fs.readFileSync(__dirname + '/../data/nmrshiftdb2.json', 'utf8'));

const molfile = `Benzene, ethyl-, ID: C100414
  NIST    16081116462D 1   1.00000     0.00000
Copyright by the U.S. Sec. Commerce on behalf of U.S.A. All rights reserved.
  8  8  0     0  0              1 V2000
    0.5015    0.0000    0.0000 C   0  0  0  0  0  0           0  0  0
    0.0000    0.8526    0.0000 C   0  0  0  0  0  0           0  0  0
    1.5046    0.0000    0.0000 C   0  0  0  0  0  0           0  0  0
    2.0062    0.8526    0.0000 C   0  0  0  0  0  0           0  0  0
    3.0092    0.8526    0.0000 C   0  0  0  0  0  0           0  0  0
    1.5046    1.7554    0.0000 C   0  0  0  0  0  0           0  0  0
    0.5015    1.7052    0.0000 C   0  0  0  0  0  0           0  0  0
    3.5108    0.0000    0.0000 C   0  0  0  0  0  0           0  0  0
  1  2  2  0     0  0
  3  1  1  0     0  0
  2  7  1  0     0  0
  4  3  2  0     0  0
  4  5  1  0     0  0
  6  4  1  0     0  0
  5  8  1  0     0  0
  7  6  2  0     0  0
M  END
`;

describe('Spinus prediction', function () {
    it('1H chemical shift prediction expanded', async function () {
        this.timeout(10000);
        const prediction = await predictor.spinus(molfile);
        prediction.length.should.equal(10);
    });
    it('1H chemical shift prediction grouped', async function () {
        this.timeout(10000);
        const prediction = await predictor.spinus(molfile, {group: true});
        //console.log(JSON.stringify(prediction));
        prediction.length.should.equal(5);
    });
    it('1H chemical shift prediction expanded from SMILES', async function () {
        this.timeout(10000);
        const prediction = await predictor.spinus('c1ccccc1');
        prediction.length.should.equal(6);
    });
    it('1H chemical shift prediction expanded from SMILES', async function () {
        this.timeout(10000);
        const prediction = await predictor.spinus('c1ccccc1CC');
        prediction.length.should.equal(10);
    });
});

describe('HOSE assignment prediction', function () {
    it('1H chemical shift prediction expanded', function () {
        const prediction = predictor.proton(molfile, {db: db1H});
        prediction[0].delta.should.greaterThan(0);
        prediction.length.should.equal(10);
    });

    it('1H chemical shift prediction grouped', function () {
        const prediction = predictor.proton(molfile, {group: true, db: db1H});
        prediction[0].delta.should.greaterThan(0);
        prediction.length.should.equal(5);
    });

    it('13C chemical shift prediction expanded', function () {
        //console.log(db13C);
        const prediction = predictor.carbon(molfile, {db: db13C});
        prediction.length.should.eql(8);
        prediction[0].delta.should.greaterThan(0);
        prediction[1].delta.should.greaterThan(0);
        prediction[2].delta.should.greaterThan(0);
        prediction[3].delta.should.greaterThan(0);
        prediction[4].delta.should.greaterThan(0);
        prediction[5].delta.should.greaterThan(0);
        prediction[6].delta.should.greaterThan(0);
        prediction[7].delta.should.greaterThan(0);

    });

    it('13C chemical shift prediction grouped', function () {
        const prediction = predictor.carbon(molfile, {group: true, db: db13C});
        prediction.length.should.eql(6);
        prediction[0].delta.should.greaterThan(0);
        prediction[1].delta.should.greaterThan(0);
        prediction[2].delta.should.greaterThan(0);
        prediction[3].delta.should.greaterThan(0);
        prediction[4].delta.should.greaterThan(0);
        prediction[5].delta.should.greaterThan(0);
    });
});