// small note on the best way to define array
// http://jsperf.com/lp-array-and-loops/2

import ArrayUtils from 'ml-array-utils';
import min from 'ml-array-min';
import max from 'ml-array-max';
import getMedian from 'ml-array-median';
import rescale from 'ml-array-rescale';
import JcampConverter from 'jcampconverter';

import JcampCreator from './jcampEncoder/JcampCreator';
import peakPicking from './peakPicking/peakPicking';

const DATACLASS_XY = 1;
const DATACLASS_PEAK = 2;

/**
 * Construct the object from the given sd object(output of the jcampconverter or brukerconverter filter)
 * @class SD
 * @param {SD} sd
 * @constructor
 */
export default class SD {
  constructor(sd) {
    this.sd = sd;
    this.activeElement = 0;
  }

  /**
   * Creates a SD instance from the given jcamp.
   * @param {string} jcamp - The jcamp string to parse from
   * @param {object} options - Jcamp parsing options
   * @param {boolean} [options.keepSpectra=true] - If set to false the spectra data points will not be stored in the instance
   * @param {RegExp} [options.keepRecordsRegExp=/^.+$/] A regular expression for metadata fields to extract from the jcamp
   * @return {SD} Return the constructed SD instance
   */
  static fromJcamp(jcamp, options = {}) {
    options = Object.assign(
      {},
      { keepSpectra: true, keepRecordsRegExp: /^.+$/ },
      options,
      { xy: true },
    );
    let spectrum = JcampConverter.convert(jcamp, options);
    return new this(spectrum);
  }

  /**
   * This function create a SD instance from xy data
   * @param {Array} x - X data.
   * @param {Array} y - Y data.
   * @param {object} options - Optional parameters
   * @return {SD} SD instance from x and y data
   */
  static fromXY(x, y, options = {}) {
    const result = {};
    result.profiling = [];
    result.logs = [];
    result.info = {};
    const spectrum = {};
    spectrum.isXYdata = true;
    spectrum.nbPoints = x.length;
    spectrum.firstX = x[0];
    spectrum.firstY = y[0];
    spectrum.lastX = x[spectrum.nbPoints - 1];
    spectrum.lastY = y[spectrum.nbPoints - 1];
    spectrum.xFactor = 1;
    spectrum.yFactor = 1;
    spectrum.xUnit = options.xUnit;
    spectrum.yUnit = options.yUnit;
    spectrum.deltaX =
      (spectrum.lastX - spectrum.firstX) / (spectrum.nbPoints - 1);
    spectrum.title = options.title || 'spectra-data from xy';
    spectrum.dataType = options.dataType;
    spectrum.data = [{ x: x, y: y }];
    result.twoD = false;
    result.spectra = [spectrum];
    return new this(result);
  }

  /**
   * This function sets the nactiveSpectrum sub-spectrum as active
   * @param {number} nactiveSpectrum index of the sub-spectrum to set as active
   */
  setActiveElement(nactiveSpectrum) {
    this.activeElement = nactiveSpectrum;
  }

  /**
   * This function returns the index of the active sub-spectrum.
   * @return {number|*}
   */
  getActiveElement() {
    return this.activeElement;
  }

  /**
   * This function returns the units of the independent dimension.
   * @return {xUnit|*|M.xUnit}
   */
  getXUnits() {
    return this.getSpectrum().xUnit;
  }

  /**
   * This function set the units of the independent dimension.
   * @param {string} units of the independent dimension.
   */
  setXUnits(units) {
    this.getSpectrum().xUnit = units;
  }
  /**
   * * This function returns the units of the dependent variable.
   * @return {yUnit|*|M.yUnit}
   */
  getYUnits() {
    return this.getSpectrum().yUnit;
  }

  /**
   * This function returns the information about the dimensions
   * @param {number} index of the tuple
   * @return {number|*}
   */
  getSpectraVariable(index) {
    return this.sd.ntuples[index];
  }

  /**
   * Return the current page
   * @param {number} index - index of spectrum
   * @return {number}
   */
  getPage(index) {
    return this.sd.spectra[index].page;
  }

  /**
   * Return the number of points in the current spectrum
   * @param {number} i of sub-spectrum
   * @return {number | *}
   */
  getNbPoints(i) {
    return this.getSpectrumData(i).y.length;
  }

  /**
   * Return the first value of the independent dimension
   * @param {i} i of sub-spectrum
   * @return {number | *}
   */
  getFirstX(i = this.activeElement) {
    return this.sd.spectra[i].firstX;
  }

  /**
   * Set the firstX for this spectrum. You have to force and update of the xAxis after!!!
   * @param {number} x - The value for firstX
   * @param {number} i sub-spectrum Default:activeSpectrum
   */
  setFirstX(x, i = this.activeElement) {
    this.sd.spectra[i].firstX = x;
  }

  /**
   * Return the last value of the direct dimension
   * @param {number} i - sub-spectrum Default:activeSpectrum
   * @return {number}
   */
  getLastX(i = this.activeElement) {
    return this.sd.spectra[i].lastX;
  }

  /**
   * Set the last value of the direct dimension. You have to force and update of the xAxis after!!!
   * @param {number} x - The value for lastX
   * @param {number} i - sub-spectrum Default:activeSpectrum
   */
  setLastX(x, i = this.activeElement) {
    this.sd.spectra[i].lastX = x;
  }

  /**
   */
  /**
   * Return the first value of the direct dimension
   * @param {number} i - sub-spectrum Default:activeSpectrum
   * @return {number}
   */
  getFirstY(i = this.activeElement) {
    return this.sd.spectra[i].firstY;
  }

  /**
   * Set the first value of the indirect dimension. Only valid for 2D spectra.
   * @param {number} y - the value of firstY
   * @param {number} i - sub-spectrum Default: activeSpectrum
   */
  setFirstY(y, i = this.activeElement) {
    this.sd.spectra[i].firstY = y;
  }

  /**
   * Return the first value of the indirect dimension. Only valid for 2D spectra.
   * @param {number} i - sub-spectrum Default: activeSpectrum
   * @return {number}
   */
  getLastY(i = this.activeElement) {
    return this.sd.spectra[i].lastY;
  }

  /**
   * Return the first value of the indirect dimension
   * @param {number} y - the value of firstY
   * @param {number} i - sub-spectrum Default:activeSpectrum
   */
  setLastY(y, i = this.activeElement) {
    this.sd.spectra[i].lastY = y;
  }

  /**
   * Set the spectrum data_class. It could be DATACLASS_PEAK=1 or DATACLASS_XY=2
   * @param {string} dataClass - data_class of the current spectra data
   */
  setDataClass(dataClass) {
    if (dataClass === DATACLASS_PEAK) {
      this.getSpectrum().isPeaktable = true;
      this.getSpectrum().isXYdata = false;
    }
    if (dataClass === DATACLASS_XY) {
      this.getSpectrum().isXYdata = true;
      this.getSpectrum().isPeaktable = false;
    }
  }

  /**
   * Is this a PEAKTABLE spectrum?
   * @return {boolean}
   */
  isDataClassPeak() {
    if (this.getSpectrum().isPeaktable) {
      return this.getSpectrum().isPeaktable;
    }
    return false;
  }

  /**
   * Is this a XY spectrum?
   * @return {*}
   */
  isDataClassXY() {
    if (this.getSpectrum().isXYdata) {
      return this.getSpectrum().isXYdata;
    }
    return false;
  }

  /**
     * Set the data type for this spectrum. It could be one of the following:
     ["INFRARED"||"IR","IV","NDNMRSPEC","NDNMRFID","NMRSPEC","NMRFID","HPLC","MASS"
     * "UV", "RAMAN" "GC"|| "GASCHROMATOGRAPH","CD"|| "DICHRO","XY","DEC"]
     * @param {string} dataType
     */
  setDataType(dataType) {
    this.getSpectrum().dataType = dataType;
  }

  /**
   * Return the dataType(see: setDataType )
   * @return {string|string|*|string}
   */
  getDataType() {
    return this.getSpectrum().dataType;
  }

  /**
   * Return the i-th sub-spectrum data in the current spectrum
   * @param {number} i - sub-spectrum Default:activeSpectrum
   * @return {object}
   */
  getSpectrumData(i = this.activeElement) {
    return this.sd.spectra[i].data[0];
  }

  /**
   * Return the i-th sub-spectra in the current spectrum
   * @param {number} i - sub-spectrum Default:activeSpectrum
   * @return {object}
   */
  getSpectrum(i = this.activeElement) {
    return this.sd.spectra[i];
  }

  /**
   * Return the amount of sub-spectra in this object
   * @return {*}
   */
  getNbSubSpectra() {
    return this.sd.spectra.length;
  }

  /**
   *  Returns an array containing the x values of the spectrum
   * @param {number} i sub-spectrum Default:activeSpectrum
   * @return {Array}
   */
  getXData(i) {
    return this.getSpectrumData(i).x;
  }

  /**
   * This function returns a double array containing the values with the intensities for the current sub-spectrum.
   * @param {number} i sub-spectrum Default:activeSpectrum
   * @return {Array}
   */
  getYData(i) {
    return this.getSpectrumData(i).y;
  }

  /**
   * Returns the x value at the specified index for the active sub-spectrum.
   * @param {number} i array index between 0 and spectrum.getNbPoints()-1
   * @return {number}
   */
  getX(i) {
    return this.getXData()[i];
  }

  /**
   * Returns the y value at the specified index for the active sub-spectrum.
   * @param {number} i array index between 0 and spectrum.getNbPoints()-1
   * @return {number}
   */
  getY(i) {
    return this.getYData()[i];
  }

  /**
   * Returns a double[2][nbPoints] where the first row contains the x values and the second row the y values.
   * @param {number} i sub-spectrum Default:activeSpectrum
   * @return {*[]}
   */
  getXYData(i = this.activeElement) {
    return [this.getXData(i), this.getYData(i)];
  }

  /**
   * Return the title of the current spectrum.
   * @param {number} i sub-spectrum Default:activeSpectrum
   * @return {*}
   */
  getTitle(i) {
    return this.getSpectrum(i).title;
  }

  /**
   * Set the title of this spectrum.
   * @param {string} newTitle The new title
   * @param {number} i sub-spectrum Default:activeSpectrum
   */
  setTitle(newTitle, i) {
    this.getSpectrum(i).title = newTitle;
  }

  /**
   * This function returns the minimal value of Y
   * @param {number} i sub-spectrum Default:activeSpectrum
   * @return {number}
   */
  getMinY(i) {
    return min(this.getYData(i));
  }

  /**
   * This function returns the maximal value of Y
   * @param {number} i sub-spectrum Default:activeSpectrum
   * @return {number}
   */
  getMaxY(i) {
    return max(this.getYData(i));
  }

  /**
   * Return the min and max value of Y
   * @param {number} i sub-spectrum Default:activeSpectrum
   * @return {{min, max}|*}
   */
  getMinMaxY(i) {
    return { min: this.getMinY(i), max: this.getMaxY(i) };
  }

  /**
   * Get the noise threshold level of the current spectrum. It uses median instead of the mean
   * @param {object} options
   * @param {number} [options.from] - lower limit in ppm to compute noise level
   * @param {number} [options.to] - upper limit in ppm to compute noise level
   * @return {number}
   */
  getNoiseLevel(options = {}) {
    let { from, to } = options;
    let data =
      from !== undefined && to !== undefined
        ? this.getVector({ from, to })
        : this.getYData();
    let median = getMedian(data);
    return median * this.getNMRPeakThreshold(this.getNucleus(1));
  }

  /**
   * Return the xValue for the given index.
   * @param {number} doublePoint
   * @return {number}
   */
  arrayPointToUnits(doublePoint) {
    return (
      this.getFirstX() -
      (doublePoint * (this.getFirstX() - this.getLastX())) /
        (this.getNbPoints() - 1)
    );
  }

  /**
   * Returns the index-value for the data array corresponding to a X-value in
   * units for the element of spectraData to which it is linked (spectraNb).
   * This method makes use of spectraData.getFirstX(), spectraData.getLastX()
   * and spectraData.getNbPoints() to derive the return value if it of data class XY
   * It performs a binary search if the spectrum is a peak table
   * @param {number} inValue - value in Units to be converted
   * @return {number} An integer representing the index value of the inValue
   */
  unitsToArrayPoint(inValue) {
    if (this.isDataClassXY()) {
      return Math.round(
        (this.getFirstX() - inValue) * (-1.0 / this.getDeltaX()),
      );
    } else if (this.isDataClassPeak()) {
      let currentArrayPoint = 0;
      let upperLimit = this.getNbPoints() - 1;
      let lowerLimit = 0;
      let midPoint;

      if (this.getFirstX() > this.getLastX()) {
        upperLimit = 0;
        lowerLimit = this.getNbPoints() - 1;

        if (inValue > this.getFirstX()) {
          return this.getNbPoints();
        }
        if (inValue < this.getLastX()) {
          return -1;
        }
      } else {
        if (inValue < this.getFirstX()) {
          return -1;
        }
        if (inValue > this.getLastX()) {
          return this.getNbPoints();
        }
      }

      while (Math.abs(upperLimit - lowerLimit) > 1) {
        midPoint = Math.round(Math.floor((upperLimit + lowerLimit) / 2));
        if (this.getX(midPoint) === inValue) {
          return midPoint;
        }
        if (this.getX(midPoint) > inValue) {
          upperLimit = midPoint;
        } else {
          lowerLimit = midPoint;
        }
      }
      currentArrayPoint = lowerLimit;
      if (
        Math.abs(this.getX(lowerLimit) - inValue) >
        Math.abs(this.getX(upperLimit) - inValue)
      ) {
        currentArrayPoint = upperLimit;
      }
      return currentArrayPoint;
    } else {
      return 0;
    }
  }

  /**
   * Returns the separation between 2 consecutive points in the frequency domain
   * @return {number}
   */
  getDeltaX() {
    return (this.getLastX() - this.getFirstX()) / (this.getNbPoints() - 1);
  }

  /**
   * This function scales the values of Y between the min and max parameters
   * @param {number} min - Minimum desired value for Y
   * @param {number} max - Maximum desired value for Y
   */
  setMinMax(min, max) {
    let y = this.getYData();
    rescale(y, { min: min, max: max, output: y });
    this.updateFirstLastY();
  }

  /**
   * This function scales the values of Y to fit the min parameter
   * @param {number} min - Minimum desired value for Y
   */
  setMin(min) {
    let y = this.getYData();
    rescale(y, { min: min, output: y, autoMinMax: true });
    this.updateFirstLastY();
  }

  /**
   * This function scales the values of Y to fit the max parameter
   * @param {number} max - Maximum desired value for Y
   */
  setMax(max) {
    let y = this.getYData();
    rescale(y, { max: max, output: y, autoMinMax: true });
    this.updateFirstLastY();
  }

  /**
   * This function shifts the values of Y
   * @param {number} value - Distance of the shift
   */
  yShift(value) {
    let y = this.getYData();
    for (let i = 0; i < y.length; i++) {
      y[i] += value;
    }
    this.updateFirstLastY(y);
  }

  /**
   * This function shift the given spectraData. After this function is applied, all the peaks in the
   * spectraData will be found at xi+globalShift
   * @param {number} globalShift - Distance of the shift for direct dimension.
   */
  shift(globalShift) {
    for (let i = 0; i < this.getNbSubSpectra(); i++) {
      this.setActiveElement(i);
      let x = this.getSpectrumData().x;
      let length = this.getNbPoints();
      for (let j = 0; j < length; j++) {
        x[j] += globalShift;
      }
      this.updateFirstLastX(x);
    }
  }

  /**
   * Update first and last values of Y data.
   * @param {Array} y - array of Y spectra data.
   */
  updateFirstLastY(y) {
    if (!Array.isArray(y)) {
      y = this.getYData();
    }
    this.setFirstY(y[0]);
    this.setLastY(y[y.length - 1]);
  }

  /**
   * Update first and last values of X data.
   * @param {Array} x - array of X spectra data.
   */
  updateFirstLastX(x) {
    if (!Array.isArray(x)) {
      x = this.getXData();
    }
    this.setFirstX(x[0]);
    this.setLastX(x[x.length - 1]);
  }
  /**
   * Fills a zone of the spectrum with the given value.
   * @param {number} from - one limit the spectrum to fill
   * @param {number} to - one limit the spectrum to fill
   * @param {number} value - value with which to fill
   */
  fill(from, to, value = 0) {
    if (from > to) {
      [from, to] = [to, from];
    }

    let currentActiveElement = this.getActiveElement();
    for (let i = 0; i < this.getNbSubSpectra(); i++) {
      this.setActiveElement(i);

      let minX = this.getFirstX();
      let maxX = this.getLastX();

      if (this.getDeltaX()) [minX, maxX] = [maxX, minX];

      if (from > maxX || to < minX) {
        return;
      }

      from = Math.max(from, minX);
      to = Math.min(to, maxX);

      let start = this.unitsToArrayPoint(from);
      let end = this.unitsToArrayPoint(to);

      if (start > end) {
        [start, end] = [end, start];
      }

      let y = this.getYData();
      for (let j = start; j <= end; j++) {
        y[j] = value;
      }
      this.updateFirstLastY();
    }
    this.setActiveElement(currentActiveElement);
  }

  /**
   * This function suppress a zone from the given spectraData within the given x range.
   * Returns a spectraData of type PEAKDATA without peaks in the given region
   * @param {number} from - one limit the spectrum to suppress
   * @param {number} to - one limit the spectrum to suppress
   */
  suppressRange(from, to) {
    this.suppressRanges([{ from, to }]);
  }

  /**
   * This function suppress a zones of the given spectraData within the given x range.
   * Returns a spectraData of type PEAKDATA without peaks in the given region
   * @param {Array} zones - Array with from-to limits of the spectrum to suppress.
   */
  suppressRanges(zones = []) {
    let currentActiveElement = this.getActiveElement();
    for (let zone of zones) {
      if (zone.active) {
        let { from, to } = zone;

        if (from === to) {
          return;
        } else if (from > to) {
          [from, to] = [to, from];
        }

        let start, end, x, y;
        for (let i = 0; i < this.getNbSubSpectra(); i++) {
          this.setActiveElement(i);

          x = this.getXData();
          y = this.getYData();

          let minX = this.getFirstX();
          let maxX = this.getLastX();

          if (this.getDeltaX()) [minX, maxX] = [maxX, minX];

          if (from > maxX || to < minX) {
            return;
          }

          from = Math.max(from, minX);
          to = Math.min(to, maxX);

          start = this.unitsToArrayPoint(from);
          end = this.unitsToArrayPoint(to);

          if (start > end) {
            [start, end] = [end, start];
          }

          y.splice(start, end - start + 1);
          x.splice(start, end - start + 1);

          this.updateFirstLastX();
          this.updateFirstLastY();
          this.setDataClass(DATACLASS_PEAK);
        }
      }
    }
    this.setActiveElement(currentActiveElement);
  }

  /**
   * This function performs a simple peak detection in a spectraData. The parameters that can be specified are:
   * Returns a two dimensional array of double specifying [x,y] of the detected peaks.
   * @option from:    Lower limit.
   * @option to:      Upper limit.
   * @option threshold: The minimum intensity to consider a peak as a signal, expressed as a percentage of the highest peak.
   * @option stdev: Number of standard deviation of the noise for the threshold calculation if a threshold is not specified.
   * @option resolution: The maximum resolution of the spectrum for considering peaks.
   * @option yInverted: Is it a Y inverted spectrum?(like an IR spectrum)
   * @option smooth: A function for smoothing the spectraData before the detection. If your are dealing with
   * experimental spectra, smoothing will make the algorithm less prune to false positives.
   */
  /*
    simplePeakPicking(parameters) {
        //@TODO implements this filter
    }
    */

  /**
   * Get the maximum peak the spectrum
   * @return {[x, y]}
   */
  getMaxPeak() {
    let y = this.getSpectraDataY();
    let max = y[0];
    let index = 0;
    for (let i = 0; i < y.length; i++) {
      if (max < y[i]) {
        max = y[i];
        index = i;
      }
    }
    return [this.getX(index), max];
  }

  /** TODO: should be modifed, this is same that getParamInt and getParam
   * Get the value of the parameter. If it is null, will set up a default value
   * @param {string} name - The parameter name
   * @param {*} defvalue - The default value
   * @return {number}
   */

  getParamDouble(name, defvalue) {
    let value = this.sd.info[name];
    if (!value) {
      value = defvalue;
    }
    return value;
  }

  /**
   * Get the string of the value of the parameter. If it is null, will set up a default value
   * @param {string} name - The parameter name
   * @param {*} defvalue - The default value
   * @return {string}
   */
  getParamString(name, defvalue) {
    let value = this.sd.info[name];
    if (!value) {
      value = defvalue;
    }
    return `${value}`;
  }

  /**
   * Get the value of the parameter
   * @param {string} name - The parameter name
   * @param {*} defvalue - The default value
   * @return {number}
   */
  getParamInt(name, defvalue) {
    let value = this.sd.info[name];
    if (!value) {
      value = defvalue;
    }
    return value;
  }

  /**
   * Get the value of the parameter
   * @param {string} name - The parameter name
   * @param {*} defvalue - The default value
   * @return {*}
   */
  getParam(name, defvalue) {
    let value = this.sd.info[name];
    if (!value) {
      value = defvalue;
    }
    return value;
  }

  /**
   * True if the spectrum.info contains the given parameter
   * @param {string} name - The parameter name
   * @return {boolean}
   */
  containsParam(name) {
    if (this.sd.info[name]) {
      return true;
    }
    return false;
  }

  /**
   * Return the y elements of the current spectrum. Same as getYData. Kept for backward compatibility.
   * @return {Array}
   */
  getSpectraDataY() {
    return this.getYData();
  }

  /**
   * Return the x elements of the current spectrum. Same as getXData. Kept for backward compatibility.
   * @return {Array}
   */
  getSpectraDataX() {
    return this.getXData();
  }

  /**
   * Update min max values of X and Y axis.
   */
  resetMinMax() {
    // TODO: Implement this function
  }

  /**
   * Set a new parameter to this spectrum
   * @param {string} name - the parameter name
   * @param {number | *} value - the parameter value
   */
  putParam(name, value) {
    this.sd.info[name] = value;
  }

  /**
   * This function returns the area under the spectrum in the given window (spectrum units)
   * @param {number} from - one limit in spectrum units
   * @param {number} to - one limit in spectrum units
   * @return {number}
   */
  getArea(from, to) {
    let i0 = this.unitsToArrayPoint(from);
    let ie = this.unitsToArrayPoint(to);
    let area = 0;

    if (i0 > ie) {
      [i0, ie] = [ie, i0];
    }

    for (let i = i0; i < ie; i++) {
      area += this.getY(i);
    }
    return area * Math.abs(this.getDeltaX());
  }

  /**
   * This function return the integral values for certains ranges at specific SD instance .
   * @param {Array} ranges - array of objects ranges
   * @param {object} options - option such as nH for normalization, if it is nH is zero the integral value returned is absolute value
   */
  updateIntegrals(ranges, options = {}) {
    ranges.forEach((range) => {
      range.integral = this.getArea(range.from, range.to);
    });
    ranges.updateIntegrals({ sum: options.nH });
  }

  /**
   * Returns a equally spaced vector within the given window.
   * @param {object} options
   * @param {number} [options.from = firstX] - one limit in spectrum units
   * @param {number} [options.to = lastX] - one limit in spectrum units
   * @param {number} [options.nbPoints] - number of points to return(!!!sometimes it is not possible to return exactly the required nbPoints)
   * @param {string} [options.variant = 'slot'] - variant of the algorithm to get equally spaced data if nbPoints is an entry.
   * @return {Array}
   */
  getVector(options = {}) {
    let { from, to, nbPoints, variant } = options;

    if (nbPoints) {
      return ArrayUtils.getEquallySpacedData(
        this.getSpectraDataX(),
        this.getSpectraDataY(),
        { from, to, numberOfPoints: nbPoints, variant },
      );
    } else {
      return this.getPointsInWindow(from, to, options);
    }
  }

  /**
   * In place modification of the data to usually reduce the size
   * This will convert the data in equally spaces X.
   * @param {object} options
   * @param {number} [options.from] - one limit in spectrum units
   * @param {number} [options.to] - one limit in spectrum units
   * @param {number} [options.nbPoints] - number of points to return(!!!sometimes it is not possible to return exactly the required nbPoints)
   * @return {this}
   */
  reduceData(options = {}) {
    if (!this.isDataClassXY()) {
      throw Error('reduceData can only apply on equidistant data');
    }

    let { from, to, nbPoints } = options;

    let currentActiveElement = this.activeElement;
    for (let i = 0; i < this.getNbSubSpectra(); i++) {
      this.setActiveElement(i);
      if (this.getXUnits().toLowerCase() !== 'hz') {
        if (options.nbPoints) {
          let x = this.getSpectraDataX();
          let y = this.getSpectraDataY();

          if (x[0] > x[1] && from < to) {
            [from, to] = [to, from];
          } else if (from > to) {
            [from, to] = [to, from];
          }

          y = ArrayUtils.getEquallySpacedData(x, y, {
            from,
            to,
            numberOfPoints: nbPoints,
          });

          let step = (to - from) / (y.length - 1);

          x = new Array(y.length).fill(from);
          for (let j = 0; j < y.length; j++) {
            x[j] += step * j;
          }

          this.sd.spectra[i].data[0].x = x;
          this.sd.spectra[i].data[0].y = y;
          this.setFirstX(x[0]);
          this.setLastX(x[x.length - 1]);
          this.sd.spectra[i].nbPoints = y.length;
        } else {
          let xyData = this.getPointsInWindow(from, to, { outputX: true });
          this.sd.spectra[i].data[0] = xyData;
          this.setFirstX(xyData.x[0]);
          this.setLastX(xyData.x[xyData.x.length - 1]);
          this.sd.spectra[i].nbPoints = xyData.y.length;
        }
      }
    }
    this.setActiveElement(currentActiveElement);
    return this;
  }

  /**
   * Returns all the point in a given window.
   * Not tested, you have to know what you are doing
   * @param {number} from - index of a limit of the desired window.
   * @param {number} to - index of a limit of the desired window
   * @param {object} options
   * @param {boolean} [options.outputX = false] - if true the output will be {x, y}.
   * @return {Array | object} - Array / {x, y} data of the desired window.
   * @private
   */
  getPointsInWindow(from, to, options = {}) {
    if (!this.isDataClassXY()) {
      throw Error('getPointsInWindow can only apply on equidistant data');
    }
    let { outputX = false } = options;

    let indexOfFrom = this.unitsToArrayPoint(from);
    let indexOfTo = this.unitsToArrayPoint(to);

    if (indexOfFrom > indexOfTo) {
      [indexOfFrom, indexOfTo] = [indexOfTo, indexOfFrom];
    }
    if (indexOfFrom >= 0 && indexOfTo <= this.getNbPoints() - 2) {
      let data = this.getSpectraDataY().slice(indexOfFrom, indexOfTo + 1);
      if (outputX) {
        let x = this.getSpectraDataX().slice(indexOfFrom, indexOfTo + 1);
        data = { x, y: data };
      }
      return data;
    } else {
      throw Error('values outside this in range');
    }
  }

  /**
   * Is it a 2D spectrum?
   * @return {boolean}
   */
  is2D() {
    if (typeof this.sd.twoD === 'undefined') {
      return false;
    }
    return this.sd.twoD;
  }

  /**
   * Set the normalization value for this spectrum
   * @param {number} value - integral value to set up
   */
  setTotalIntegral(value) {
    this.totalIntegralValue = value;
  }

  /**
   * Return the normalization value. It is not set check the molfile and guess it from the number of atoms.
   * @return {number}
   */
  get totalIntegral() {
    if (this.totalIntegralValue) {
      return this.totalIntegralValue;
    } else if (this.molecule) {
      if (this.getNucleus(0).indexOf('H')) {
        return this.mf.replace(/.*H([0-9]+).*/, '$1') * 1;
      }
      if (this.getNucleus(0).indexOf('C')) {
        return this.mf.replace(/.*C([0-9]+).*/, '$1') * 1;
      }
    } else {
      return 100;
    }
    return 1;
  }

  /**
   * this function set a molfile, molecule and molecular formula.
   * @param {string} molfile - The molfile that correspond to current spectra data
   */
  setMolfile(molfile) {
    this.molfile = molfile;
  }

  setMF(mf) {
    this.mf = mf;
  }

  /**
   * this function create a new peakPicking
   * @param {object} options - parameters to calculation of peakPicking
   * @return {*}
   */
  createPeaks(options = {}) {
    this.peaks = peakPicking(this, options);
    return this.peaks;
  }

  /**
   * this function return the peak table or extract the peak of the spectrum.
   * @param {object} options - parameters to calculation of peakPicking
   * @return {*}
   */
  getPeaks(options) {
    let peaks;
    if (this.peaks) {
      peaks = this.peaks;
    } else {
      peaks = peakPicking(this, options);
    }
    return peaks;
  }

  /* autoAssignment(options) {

    }*/

  /**
   * This function creates a String that represents the given spectraData in the format JCAMP-DX 5.0
   * The X,Y data can be compressed using one of the methods described in:
   * "JCAMP-DX. A STANDARD FORMAT FOR THE EXCHANGE OF ION MOBILITY SPECTROMETRY DATA",
   *  http://www.iupac.org/publications/pac/pdf/2001/pdf/7311x1765.pdf
   * @param {object} options - some options are availables:
   * @option {string} encode  - ['FIX','SQZ','DIF','DIFDUP','CVS','PAC'] (Default: 'DIFDUP')
   * @option {number} yfactor - The YFACTOR. It allows to compress the data by removing digits from the ordinate. (Default: 1)
   * @option {string} type - ["NTUPLES", "SIMPLE"] (Default: "SIMPLE")
   * @option {object} keep - A set of user defined parameters of the given SpectraData to be stored in the jcamp.
   * @example SD.toJcamp(spectraData,{encode:'DIFDUP',yfactor:0.01,type:"SIMPLE",keep:['#batchID','#url']});
   * @return {*} a string containing the jcamp-DX file
   */
  toJcamp(options = {}) {
    let creator = new JcampCreator();
    return creator.convert(
      this,
      Object.assign(
        {},
        { yFactor: 1, encode: 'DIFDUP', type: 'SIMPLE' },
        options,
      ),
    );
  }
}
