type IntLiteral = -1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 60;
type int = number & ({__type__: 'int'} | IntLiteral);

const int = (x: number): int => (x | 0) as int;

const assert = (x: boolean, message?: () => string) => {
  if (x) return;
  throw new Error(message ? message() : 'Assertion failed!');
};

const nonnull = <T>(x: T | null, message?: () => string): T => {
  if (x !== null) return x;
  throw new Error(message ? message() : 'Unexpected null!');
};

const only = <T>(xs: T[]): T => {
  assert(xs.length === 1);
  return xs[0];
};

//////////////////////////////////////////////////////////////////////////////

declare const require: any;

interface Point { x: int, y: int };
interface Bound { min: Point, max: Point };

type AnimData = {
  name: string,
  width: int,
  height: int,
};

type PNGData = {
  width: int,
  height: int,
  channels: int,
  data: Uint8Array,
};

const show = (point: Point): string => {
  return `(${point.x}, ${point.y})`;
};

const showFrame = (frame: Point): string => {
  return `Frame (${frame.y}, ${frame.x})`;
};

const findBounds = (anim: AnimData, frame: Point, sprite: PNGData): Bound => {
  const result = {
    min: {x: anim.width, y: anim.height},
    max: {x: int(0), y: int(0)},
  };
  for (let x = 0; x < anim.width; x++) {
    for (let y = 0; y < anim.height; y++) {
      const index = (frame.x * anim.width + x) +
                    (frame.y * anim.height + y) * sprite.width;
      if (sprite.data[4 * index + 3] === 0) continue;
      result.min.x = int(Math.min(result.min.x, x));
      result.min.y = int(Math.min(result.min.y, y));
      result.max.x = int(Math.max(result.max.x, x + 1));
      result.max.y = int(Math.max(result.max.y, y + 1));
    }
  }
  return result;
};

const findOrigin = (anim: AnimData, frame: Point, shadow: PNGData): Point => {
  const result: Point[] = [];
  for (let x = 0; x < anim.width; x++) {
    for (let y = 0; y < anim.height; y++) {
      const index = (frame.x * anim.width + x) +
                    (frame.y * anim.height + y) * shadow.width;
      const r = shadow.data[4 * index + 0];
      const g = shadow.data[4 * index + 1];
      const b = shadow.data[4 * index + 2];
      if (r === 255 && g === 255 && b === 255) {
        result.push({x: int(x), y: int(y)});
      }
    }
  }
  if (result.length !== 1) {
    const error = result.length === 0 ? 'no' : 'multiple';
    throw new Error(`${showFrame(frame)}: ${error} origins found!`);
  }
  return only(result);
};

const main = (anim: AnimData, sprite: PNGData, shadow: PNGData) => {
  assert(sprite.channels === 4);
  assert(shadow.channels === 4);
  assert(sprite.width % anim.width === 0);
  assert(sprite.height % anim.height === 0);
  assert(sprite.width === shadow.width);
  assert(sprite.height === shadow.height);

  const rows = sprite.height / anim.height;
  const cols = sprite.width / anim.width;
  console.log(`Size: (${anim.width}, ${anim.height})`);
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const frame = {x: int(col), y: int(row)};
      const bound = findBounds(anim, frame, sprite);
      const point = findOrigin(anim, frame, shadow);
      console.log(`${showFrame(frame)}: center: (${point.x}, ${point.y}); ` +
                  `min: ${show(bound.min)}; max: ${show(bound.max)}`);
    }
  }
};

//////////////////////////////////////////////////////////////////////////////

const fs = require('fs');
const pngparse = require('../lib/pngparse');
const xml_parser = require('../lib/xml-parser');

interface XML {
  tagName: string,
  innerXML: string,
  childNodes?: XML[],
};

const onlyChild = (xml: XML, tag: string): XML => {
  return only(xml.childNodes!.filter(x => x.tagName === tag));
};

const parseXML = (filename: string): AnimData[] => {
  const buffer = fs.readFileSync(filename);
  const xml = xml_parser.parseFromString(buffer.toString())[2] as XML;
  assert(xml.tagName === 'AnimData');

  const result = [];
  const anim = onlyChild(xml, 'Anims');
  const anims = anim.childNodes!.filter(x => x.tagName === 'Anim');
  for (const anim of anims) {
    const name = onlyChild(anim, 'Name').innerXML;
    if (anim.childNodes!.some(x => x.tagName === 'CopyOf')) continue;
    const width  = int(parseInt(onlyChild(anim, 'FrameWidth').innerXML, 10));
    const height = int(parseInt(onlyChild(anim, 'FrameHeight').innerXML, 10));
    result.push({name, width, height});
  }
  return result;
};

const parsePNG = (filename: string): Promise<PNGData> => {
  return new Promise((resolve, reject) => {
    pngparse.parseFile(filename, (error: Error, data: PNGData) => {
      error ? reject(error) : resolve(data);
    });
  });
};

const load = async (root: string, type: string) => {
  const anims = parseXML(`${root}/AnimData.xml`);
  const walk = only(anims.filter(x => x.name === type));
  const name = `${root}/${type}-Anim.png`;

  const images = await Promise.all([
    parsePNG(`${root}/${type}-Anim.png`),
    parsePNG(`${root}/${type}-Shadow.png`),
  ]);
  main(walk, images[0], images[1]);
};

interface Process { argv: string[] };
declare const process: Process;

let index = `${parseInt(process.argv[2], 10)}`;
while (index.length < 4) index = `0${index}`;
load(`../Examples/SpriteCollab/sprite/${index}`, process.argv[3]);
