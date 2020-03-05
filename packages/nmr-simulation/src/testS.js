const { Matrix, EVD } = require('ml-matrix');
const SparseMatrix = require('ml-sparse-matrix');
const binarySearch = require('binary-search');
const { asc: sortAsc } = require('num-sort');
const spectrumGenerator = require('spectrum-generator');

const newArray = require('new-array');
const simpleClustering = require('ml-simple-clustering');
const hlClust = require('ml-hclust');

const pauli2 = createPauli(2);
const smallValue = 1e-2;

let options1h = {
    frequency: 400.082470657773,
    from: 0,
    to: 11,
    lineWidth: 1,
    nbPoints: 1024,
    maxClusterSize: 8,
    output: 'xy',
  };

class SpinSystem {
    constructor(chemicalShifts, couplingConstants, multiplicity) {
      this.chemicalShifts = chemicalShifts;
      this.couplingConstants = couplingConstants;
      this.multiplicity = multiplicity;
      this.nSpins = chemicalShifts.length;
      this._initConnectivity();
      this._initClusters();
    }
  
    static fromSpinusPrediction(result) {
      let lines = result.split('\n');
      let nspins = lines.length - 1;
      let cs = new Array(nspins);
      let integrals = new Array(nspins);
      let ids = {};
      let jc = Matrix.zeros(nspins, nspins);
      for (let i = 0; i < nspins; i++) {
        var tokens = lines[i].split('\t');
        cs[i] = Number(tokens[2]);
        ids[tokens[0] - 1] = i;
        integrals[i] = Number(tokens[5]); // Is it always 1??
      }
      for (let i = 0; i < nspins; i++) {
        tokens = lines[i].split('\t');
        let nCoup = (tokens.length - 4) / 3;
        for (j = 0; j < nCoup; j++) {
          let withID = tokens[4 + 3 * j] - 1;
          let idx = ids[withID];
          // jc[i][idx] = +tokens[6 + 3 * j];
          jc.set(i, idx, Number(tokens[6 + 3 * j]));
        }
      }
  
      for (var j = 0; j < nspins; j++) {
        for (let i = j; i < nspins; i++) {
          jc.set(j, i, jc.get(i, j));
        }
      }
      return new SpinSystem(cs, jc, newArray(nspins, 2));
    }
  
    static fromPrediction(input) {
      let predictions = SpinSystem.ungroupAtoms(input);
      const nSpins = predictions.length;
      const cs = new Array(nSpins);
      const jc = Matrix.zeros(nSpins, nSpins);
      const multiplicity = new Array(nSpins);
      const ids = {};
      let i, k, j;
      for (i = 0; i < nSpins; i++) {
        cs[i] = predictions[i].delta;
        ids[predictions[i].atomIDs[0]] = i;
      }
      for (i = 0; i < nSpins; i++) {
        cs[i] = predictions[i].delta;
        j = predictions[i].j;
        for (k = 0; k < j.length; k++) {
          // jc[ids[predictions[i].atomIDs[0]]][ids[j[k].assignment]] = j[k].coupling;
          // jc[ids[j[k].assignment]][ids[predictions[i].atomIDs[0]]] = j[k].coupling;
          jc.set(
            ids[predictions[i].atomIDs[0]],
            ids[j[k].assignment],
            j[k].coupling,
          );
          jc.set(
            ids[j[k].assignment],
            ids[predictions[i].atomIDs[0]],
            j[k].coupling,
          );
        }
        multiplicity[i] = predictions[i].integral + 1;
      }
      return new SpinSystem(cs, jc, multiplicity);
    }
  
    static ungroupAtoms(prediction) {
      let result = [];
      prediction.forEach((pred) => {
        let atomIDs = pred.atomIDs;
        if (atomIDs instanceof Array) {
          for (let i = 0; i < atomIDs.length; i++) {
            let tempPred = JSON.parse(JSON.stringify(pred));
            let nmrJ = [];
            tempPred.atomIDs = [atomIDs[i]];
            tempPred.integral = 1;
            if (tempPred.j instanceof Array) {
              for (let j = 0; j < tempPred.j.length; j++) {
                let assignment = tempPred.j[j].assignment;
                if (assignment instanceof Array) {
                  for (let k = 0; k < assignment.length; k++) {
                    let tempJ = JSON.parse(JSON.stringify(tempPred.j[j]));
                    tempJ.assignment = assignment[k];
                    nmrJ.push(tempJ);
                  }
                }
              }
            }
            tempPred.j = nmrJ;
            delete tempPred.nbAtoms;
            result.push(tempPred);
          }
        }
      });
  
      return result;
    }
  
    _initClusters() {
      this.clusters = simpleClustering(this.connectivity.to2DArray(), {
        out: 'indexes',
      });
    }
  
    //I am asumming that couplingConstants is a square matrix
    _initConnectivity() {
      const couplings = this.couplingConstants;
      const connectivity = Matrix.ones(couplings.rows, couplings.rows);
      for (let i = 0; i < couplings.rows; i++) {
        for (let j = i; j < couplings.columns; j++) {
          if (couplings.get(i, j) === 0) {
            connectivity.set(i, j, 0);
            connectivity.set(j, i, 0);
          }
        }
      }
      this.connectivity = connectivity;
    }
  
    _calculateBetas(J, frequency) {
      let betas = Matrix.zeros(J.rows, J.rows);
      // Before clustering, we must add hidden J, we could use molecular information if available
      let i, j;
      for (i = 0; i < J.rows; i++) {
        for (j = i; j < J.columns; j++) {
          let element = J.get(i, j);
          if (this.chemicalShifts[i] - this.chemicalShifts[j] !== 0) {
            let value =
              1 -
              Math.abs(
                element /
                  ((this.chemicalShifts[i] - this.chemicalShifts[j]) * frequency),
              );
            betas.set(i, j, value);
            betas.set(j, i, value);
          } else if (!(i === j || element !== 0)) {
            betas.set(i, j, 1);
            betas.set(j, i, 1);
          }
        }
      }
      return betas;
    }
  
    ensureClusterSize(options) {
      let betas = this._calculateBetas(
        this.couplingConstants,
        options.frequency || 400,
      );
      let cluster = hlClust.agnes(betas.to2DArray(), { isDistanceMatrix: true });
      let list = [];
      this._splitCluster(cluster, list, options.maxClusterSize || 8, false);
      let clusters = this._mergeClusters(list);
      this.nClusters = clusters.length;
      this.clusters = new Array(clusters.length);
  
      for (let j = 0; j < this.nClusters; j++) {
        this.clusters[j] = [];
        for (let i = 0; i < this.nSpins; i++) {
          let element = clusters[j][i];
          if (element !== 0) {
            if (element < 0) {
              this.clusters[j].push(-(i + 1));
            } else {
              this.clusters[j].push(i);
            }
          }
        }
      }
    }
  
    /**
     * Recursively split the clusters until the maxClusterSize criteria has been ensured.
     * @param {Array} cluster
     * @param {Array} clusterList
     * @param {number} maxClusterSize
     * @param  {boolean} force
     */
    _splitCluster(cluster, clusterList, maxClusterSize, force) {
      if (!force && cluster.index.length <= maxClusterSize) {
        clusterList.push(this._getMembers(cluster));
      } else {
        for (let child of cluster.children) {
          if (!isNaN(child.index) || child.index.length <= maxClusterSize) {
            let members = this._getMembers(child);
            // Add the neighbors that shares at least 1 coupling with the given cluster
            let count = 0;
            for (let i = 0; i < this.nSpins; i++) {
              if (members[i] === 1) {
                count++;
                for (let j = 0; j < this.nSpins; j++) {
                  if (this.connectivity.get(i, j) === 1 && members[j] === 0) {
                    members[j] = -1;
                    count++;
                  }
                }
              }
            }
  
            if (count <= maxClusterSize) {
              clusterList.push(members);
            } else {
              if (isNaN(child.index)) {
                this._splitCluster(child, clusterList, maxClusterSize, true);
              } else {
                // We have to threat this spin alone and use the resurrection algorithm instead of the simulation
                members[child.index] = 2;
                clusterList.push(members);
              }
            }
          } else {
            this._splitCluster(child, clusterList, maxClusterSize, false);
          }
        }
      }
    }
    /**
     * Recursively gets the cluster members
     * @param cluster
     * @param members
     */
  
    _getMembers(cluster) {
      let members = new Array(this.nSpins);
      for (let i = 0; i < this.nSpins; i++) {
        members[i] = 0;
      }
      if (!isNaN(cluster.index)) {
        members[cluster.index * 1] = 1;
      } else {
        for (let index of cluster.index) {
          members[index.index * 1] = 1;
        }
      }
      return members;
    }
  
    _mergeClusters(list) {
      let nElements = 0;
      let clusterA, clusterB, i, j, index, common, count;
      for (i = list.length - 1; i >= 0; i--) {
        clusterA = list[i];
        nElements = clusterA.length;
        index = 0;
  
        // Is it a candidate to be merged?
        while (index < nElements && clusterA[index++] !== -1);
  
        if (index < nElements) {
          for (j = list.length - 1; j >= i + 1; j--) {
            clusterB = list[j];
            // Do they have common elements?
            index = 0;
            common = 0;
            count = 0;
            while (index < nElements) {
              if (clusterA[index] * clusterB[index] === -1) {
                common++;
              }
              if (clusterA[index] !== 0 || clusterB[index] !== 0) {
                count++;
              }
              index++;
            }
  
            if (common > 0 && count <= this.maxClusterSize) {
              // Then we can merge those 2 clusters
              index = 0;
              while (index < nElements) {
                if (clusterB[index] === 1) {
                  clusterA[index] = 1;
                } else {
                  if (clusterB[index] === -1 && clusterA[index] !== 1) {
                    clusterA[index] = -1;
                  }
                }
                index++;
              }
              // list.remove(clusterB);
              list.splice(j, 1);
              j++;
            }
          }
        }
      }
  
      return list;
    }
  }


let prediction1h = [
    {
      atomIDs: ['15', '16', '17'],
      diaIDs: ['did@`@fTeYWaj@@@GzP`HeT'],
      nbAtoms: 3,
      delta: 0.992,
      atomLabel: 'H',
      multiplicity: 't',
    },
    {
      atomIDs: ['9'],
      diaIDs: ['did@`@fTfUvf`@h@GzP`HeT'],
      nbAtoms: 1,
      delta: 7.196,
      atomLabel: 'H',
      multiplicity: 'tt',
    },
    {
      atomIDs: ['10', '13'],
      diaIDs: ['did@`@fTfYUn`HH@GzP`HeT'],
      nbAtoms: 2,
      delta: 7.162,
      atomLabel: 'H',
      multiplicity: 'dddd',
    },
    {
      atomIDs: ['11', '12'],
      diaIDs: ['did@`@fTf[Waj@@bJ@_iB@bUP'],
      nbAtoms: 2,
      delta: 2.653,
      atomLabel: 'H',
      multiplicity: 'q',
    },
    {
      atomIDs: ['8', '14'],
      diaIDs: ['did@`@f\\bbRaih@J@A~dHBIU@'],
      nbAtoms: 2,
      delta: 7.26,
      atomLabel: 'H',
      multiplicity: 'tdd',
    },
  ];
  
const sp = SpinSystem.fromPrediction(prediction1h);

sp.ensureClusterSize(options1h);
let simulation = simulate1d(sp, options1h);
// console.log(simulation)
return
function simulate1d(spinSystem, options) {
    let i, j;
    let {
      lineWidth = 1,
      nbPoints = 1024,
      maxClusterSize = 10,
      output = 'y',
      frequency: frequencyMHz = 400,
      noiseFactor = 1,
      lortogauRatio = 0.5
    } = options;
  
    nbPoints = Number(nbPoints);
  
    const from = options.from * frequencyMHz || 0;
    const to = (options.to || 10) * frequencyMHz;
  
    const chemicalShifts = spinSystem.chemicalShifts.slice();
    for (i = 0; i < chemicalShifts.length; i++) {
      chemicalShifts[i] = chemicalShifts[i] * frequencyMHz;
    }
  
    // Prepare pseudo voigt
    let lineWidthPointsG = lortogauRatio * (nbPoints * lineWidth) / Math.abs(to - from) / 2.355;
    let lineWidthPointsL = (1 - lortogauRatio) * (nbPoints * lineWidth) / Math.abs(to - from) / 2;
    let lnPoints = lineWidthPointsL * 40;
  
    const gaussianLength = lnPoints | 0;
    const gaussian = new Array(gaussianLength);
    const b = lnPoints / 2;
    const c = lineWidthPointsG * lineWidthPointsG * 2;
    const l2 = lineWidthPointsL * lineWidthPointsL;
    const g2pi = lineWidthPointsG * Math.sqrt(2 * Math.PI);
    for (i = 0; i < gaussianLength; i++) {
      let x2 = (i - b) * (i - b);
      gaussian[i] =
        10e9 *
        (Math.exp(-x2 / c) / g2pi + lineWidthPointsL / ((x2 + l2) * Math.PI));
    }
  
    let result = options.withNoise
      ? [...new Array(nbPoints)].map(() => Math.random() * noiseFactor)
      : new Array(nbPoints).fill(0);
    
      let peaks = [];
    const multiplicity = spinSystem.multiplicity;
    for (let h = 0; h < spinSystem.clusters.length; h++) {
      const cluster = spinSystem.clusters[h];
  
      let clusterFake = new Array(cluster.length);
      for (i = 0; i < cluster.length; i++) {
        clusterFake[i] = cluster[i] < 0 ? -cluster[i] - 1 : cluster[i];
      }
  
      let weight = 1;
      var sumI = 0;
      const frequencies = [];
      const intensities = [];
      if (cluster.length > maxClusterSize) {
        // This is a single spin, but the cluster exceeds the maxClusterSize criteria
        // we use the simple multiplicity algorithm
        // Add the central peak. It will be split with every single J coupling.
        let index = 0;
        while (cluster[index++] < 0);
        index = cluster[index - 1];
        var currentSize, jc;
        frequencies.push(-chemicalShifts[index]);
        for (i = 0; i < cluster.length; i++) {
          if (cluster[i] < 0) {
            jc = spinSystem.couplingConstants.get(index, clusterFake[i]) / 2;
            currentSize = frequencies.length;
            for (j = 0; j < currentSize; j++) {
              frequencies.push(frequencies[j] + jc);
              frequencies[j] -= jc;
            }
          }
        }
  
        frequencies.sort(sortAsc);
        sumI = frequencies.length;
        weight = 1;
  
        for (i = 0; i < sumI; i++) {
          intensities.push(1);
        }
      } else {
        const hamiltonian = getHamiltonian(
          chemicalShifts,
          spinSystem.couplingConstants,
          multiplicity,
          spinSystem.connectivity,
          clusterFake,
        );
        const hamSize = hamiltonian.rows;
        const evd = new EVD(hamiltonian);
        const V = evd.eigenvectorMatrix;
        const diagB = evd.realEigenvalues;
        const assignmentMatrix = new SparseMatrix(hamSize, hamSize);
        const multLen = cluster.length;
        weight = 0;
        for (let n = 0; n < multLen; n++) {
          const L = getPauli(multiplicity[clusterFake[n]]);
  
          let temp = 1;
          for (j = 0; j < n; j++) {
            temp *= multiplicity[clusterFake[j]];
          }
          const A = SparseMatrix.eye(temp);
  
          temp = 1;
          for (j = n + 1; j < multLen; j++) {
            temp *= multiplicity[clusterFake[j]];
          }
          const B = SparseMatrix.eye(temp);
          const tempMat = A.kroneckerProduct(L.m).kroneckerProduct(B);
          if (cluster[n] >= 0) {
            assignmentMatrix.add(tempMat.mul(cluster[n] + 1));
            weight++;
          } else {
            assignmentMatrix.add(tempMat.mul(cluster[n]));
          }
        }
  
        let rhoip = Matrix.zeros(hamSize, hamSize);
        assignmentMatrix.forEachNonZero((i, j, v) => {
          if (v > 0) {
            for (let k = 0; k < V.columns; k++) {
              let element = V.get(j, k);
              if (element !== 0) {
                rhoip.set(i, k, rhoip.get(i, k) + element);
              }
            }
          }
          return v;
        });
  
        let rhoip2 = rhoip.clone();
        assignmentMatrix.forEachNonZero((i, j, v) => {
          if (v < 0) {
            for (let k = 0; k < V.columns; k++) {
              let element = V.get(j, k);
              if (element !== 0) {
                rhoip2.set(i, k, rhoip2.get(i, k) + element);
              }
            }
          }
          return v;
        });
        const tV = V.transpose();
  
        rhoip = tV.mmul(rhoip);
        rhoip = new SparseMatrix(rhoip.to2DArray(), { threshold: smallValue });
        triuTimesAbs(rhoip, smallValue);
        rhoip2 = tV.mmul(rhoip2);
  
        rhoip2 = new SparseMatrix(rhoip2.to2DArray(), { threshold: smallValue });
        rhoip2.forEachNonZero((i, j, v) => {
          return v;
        });
        triuTimesAbs(rhoip2, smallValue);
        // eslint-disable-next-line no-loop-func
        rhoip2.forEachNonZero((i, j, v) => {
          let val = rhoip.get(i, j);
          val = Math.min(Math.abs(val), Math.abs(v));
          val *= val;
  
          sumI += val;
          let valFreq = diagB[i] - diagB[j];
          let insertIn = binarySearch(frequencies, valFreq, sortAsc);
          if (insertIn < 0) {
            frequencies.splice(-1 - insertIn, 0, valFreq);
            intensities.splice(-1 - insertIn, 0, val);
          } else {
            intensities[insertIn] += val;
          }
        });
      }
      const numFreq = frequencies.length;
      console.log(frequencies, intensities)
      if (numFreq > 0) {
        weight = weight / sumI;
        const diff = lineWidth / 64;
        let valFreq = frequencies[0];
        let inte = intensities[0];
        let count = 1;
        for (i = 1; i < numFreq; i++) {
          if (Math.abs(frequencies[i] - valFreq / count) < diff) {
            inte += intensities[i];
            valFreq += frequencies[i];
            count++;
          } else {
            let freq = valFreq / count; 
            let center = ((nbPoints * (-freq - from)) / (to - from)) | 0;
            peaks.push([center, inte * weight]);
            valFreq = frequencies[i];
            inte = intensities[i];
            count = 1;
          }
        }
        let freq = valFreq / count;
        let center = ((nbPoints * (-freq - from)) / (to - from)) | 0;
        peaks.push([center, inte * weight])
      }
    if (numFreq > 0) {
        weight = weight / sumI;
        const diff = lineWidth / 64;
        let valFreq = frequencies[0];
        let inte = intensities[0];
        let count = 1;
        for (i = 1; i < numFreq; i++) {
          if (Math.abs(frequencies[i] - valFreq / count) < diff) {
            inte += intensities[i];
            valFreq += frequencies[i];
            count++;
          } else {
            addPeak(
              result,
              valFreq / count,
              inte * weight,
              from,
              to,
              nbPoints,
              gaussian,
            );
            valFreq = frequencies[i];
            inte = intensities[i];
            count = 1;
          }
        }
        addPeak(
          result,
          valFreq / count,
          inte * weight,
          from,
          to,
          nbPoints,
          gaussian,
        );
      }
    }
    peaks[0][0] = 5;
    console.log(peaks)
    console.log('linewidth', (nbPoints) / Math.abs(to - from) * frequencyMHz)
    let g = spectrumGenerator.generateSpectrum(peaks, {
      start: 0,
      end: nbPoints,
      pointsPerUnit: 1,
      peakWidthFct: () => 1,
      maxSize: 1e7,
      shape: {
        kind: 'gaussian',
        options: {
          fwhm: 5,
          length: nbPoints
        },
      },
    });
    console.log(peaks);
    console.log(JSON.stringify(g))
    console.log(g.y.length)
    console.log(nbPoints)
    console.log(JSON.stringify(result))
      
    if (output === 'xy') {
      return { x: _getX(options.from, options.to, nbPoints), y: result };
    }
    if (output === 'y') {
      return result;
    }
    throw new RangeError('wrong output option');
  }
  function addPeak(result, freq, height, from, to, nbPoints, gaussian) {
    const center = ((nbPoints * (-freq - from)) / (to - from)) | 0;
    console.log(center)
    const lnPoints = gaussian.length;
    let index = 0;
    let indexLorentz = 0;
    for (let i = center - lnPoints / 2; i < center + lnPoints / 2; i++) {
      index = i | 0;
      if (i >= 0 && i < nbPoints) {
        result[index] = result[index] + gaussian[indexLorentz] * height;
      }
      indexLorentz++;
    }
  }
  
  function triuTimesAbs(A, val) {
    A.forEachNonZero((i, j, v) => {
      if (i > j) return 0;
      if (Math.abs(v) <= val) return 0;
      return v;
    });
  }

  function getHamiltonian(
    chemicalShifts,
    couplingConstants,
    multiplicity,
    conMatrix,
    cluster,
  ) {
    let hamSize = 1;
    for (let i = 0; i < cluster.length; i++) {
      hamSize *= multiplicity[cluster[i]];
    }
  
    const clusterHam = new SparseMatrix(hamSize, hamSize);
  
    for (let pos = 0; pos < cluster.length; pos++) {
      let n = cluster[pos];
  
      const L = getPauli(multiplicity[n]);
  
      let A1, B1;
      let temp = 1;
      for (let i = 0; i < pos; i++) {
        temp *= multiplicity[cluster[i]];
      }
      A1 = SparseMatrix.eye(temp);
  
      temp = 1;
      for (let i = pos + 1; i < cluster.length; i++) {
        temp *= multiplicity[cluster[i]];
      }
      B1 = SparseMatrix.eye(temp);
  
      const alpha = chemicalShifts[n];
      const kronProd = A1.kroneckerProduct(L.z).kroneckerProduct(B1);
      clusterHam.add(kronProd.mul(alpha));
      for (let pos2 = 0; pos2 < cluster.length; pos2++) {
        const k = cluster[pos2];
        if (conMatrix.get(n, k) === 1) {
          const S = getPauli(multiplicity[k]);
  
          let A2, B2;
          let temp = 1;
          for (let i = 0; i < pos2; i++) {
            temp *= multiplicity[cluster[i]];
          }
          A2 = SparseMatrix.eye(temp);
  
          temp = 1;
          for (let i = pos2 + 1; i < cluster.length; i++) {
            temp *= multiplicity[cluster[i]];
          }
          B2 = SparseMatrix.eye(temp);
  
          const kron1 = A1.kroneckerProduct(L.x)
            .kroneckerProduct(B1)
            .mmul(A2.kroneckerProduct(S.x).kroneckerProduct(B2));
          kron1.add(
            A1.kroneckerProduct(L.y)
              .kroneckerProduct(B1)
              .mul(-1)
              .mmul(A2.kroneckerProduct(S.y).kroneckerProduct(B2)),
          );
          kron1.add(
            A1.kroneckerProduct(L.z)
              .kroneckerProduct(B1)
              .mmul(A2.kroneckerProduct(S.z).kroneckerProduct(B2)),
          );
  
          clusterHam.add(kron1.mul(couplingConstants.get(n, k) / 2));
        }
      }
    }
    return clusterHam;
  }
  
  function _getX(from, to, nbPoints) {
    const x = new Array(nbPoints);
    const dx = (to - from) / (nbPoints - 1);
    for (let i = 0; i < nbPoints; i++) {
      x[i] = from + i * dx;
    }
    return x;
  }

  function createPauli(mult) {
    const spin = (mult - 1) / 2;
    const prjs = new Array(mult);
    const temp = new Array(mult);
    for (var i = 0; i < mult; i++) {
      prjs[i] = mult - 1 - i - spin;
      temp[i] = Math.sqrt(spin * (spin + 1) - prjs[i] * (prjs[i] + 1));
    }
    const p = diag(temp, 1, mult, mult);
    for (i = 0; i < mult; i++) {
      temp[i] = Math.sqrt(spin * (spin + 1) - prjs[i] * (prjs[i] - 1));
    }
    const m = diag(temp, -1, mult, mult);
    const x = p
      .clone()
      .add(m)
      .mul(0.5);
    const y = m
      .clone()
      .mul(-1)
      .add(p)
      .mul(-0.5);
    const z = diag(prjs, 0, mult, mult);
    return { x, y, z, m, p };
  }
  
  function diag(A, d, n, m) {
    const diag = new SparseMatrix(n, m, { initialCapacity: 20 });
    for (let i = 0; i < A.length; i++) {
      if (i - d >= 0 && i - d < n && i < m) {
        diag.set(i - d, i, A[i]);
      }
    }
    return diag;
  }
  
  function getPauli(mult) {
    if (mult === 2) return pauli2;
    else return createPauli(mult);
  }

  