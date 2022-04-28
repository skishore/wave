import {int, Tensor3, Vec3} from './base.js';
import {Mesh, Renderer} from './renderer.js';

//////////////////////////////////////////////////////////////////////////////

type BlockId = int & {__type__: 'BlockId'};
type MaterialId = int & {__type__: 'MaterialId'};

const kNoMaterial = 0 as MaterialId;
const kEmptyBlock = 0 as BlockId;

interface Material {
  color: [number, number, number, number],
  texture: string | null,
  textureIndex: int,
};

interface Registry {
  _solid: boolean[];
  _opaque: boolean[];
  getBlockFaceMaterial(id: BlockId, face: int): MaterialId;
  getMaterialData(id: MaterialId): Material;
};

//////////////////////////////////////////////////////////////////////////////

interface GeometryData {
  numQuads: int;
  quadMaterials: MaterialId[]; // length: n = numQuads
  positions: number[];         // length: 12n (Vec3 for each vertex)
  normals: number[];           // length: 12n (Vec3 for each vertex)
  indices: int[];              // length: 6n  (2 triangles - 6 indices)
  colors: number[];            // length: 16n (Color4 for each vertex)
  uvws: number[];              // length: 12n ((u, v, w) for each vertex)
};

const kGeometryData: GeometryData = {
  numQuads: 0,
  quadMaterials: [],
  positions: [],
  normals: [],
  indices: [],
  colors: [],
  uvws: [],
};

const kTmpPos = Vec3.create();
let kMaskData = new Int16Array();

const kIndexOffsets = {
  A: [0, 1, 2, 0, 2, 3],
  B: [1, 2, 3, 0, 1, 3],
  C: [0, 2, 1, 0, 3, 2],
  D: [3, 1, 0, 3, 2, 1],
};

class TerrainMesher {
  solid: boolean[];
  opaque: boolean[];
  getBlockFaceMaterial: (id: BlockId, face: int) => MaterialId;
  getMaterialData: (id: MaterialId) => Material;
  renderer: Renderer;

  constructor(registry: Registry, renderer: Renderer) {
    this.solid = registry._solid;
    this.opaque = registry._opaque;
    this.getBlockFaceMaterial = registry.getBlockFaceMaterial.bind(registry);
    this.getMaterialData = registry.getMaterialData.bind(registry);
    this.renderer = renderer;
  }

  mesh(voxels: Tensor3): Mesh | null {
    const data = this.computeGeometryData(voxels);
    const numQuads = data.numQuads;
    if (data.numQuads === 0) return null;

    const geo = {
      positions : new Float32Array(numQuads * 12),
      normals   : new Float32Array(numQuads * 12),
      indices   : new   Uint32Array(numQuads * 6),
      colors    : new Float32Array(numQuads * 16),
      uvws      : new Float32Array(numQuads * 12),
    };

    this.copyFloats(geo.positions, data.positions);
    this.copyFloats(geo.normals,   data.normals);
    this.copyInt32s(geo.indices,   data.indices);
    this.copyFloats(geo.colors,    data.colors);
    this.copyFloats(geo.uvws,      data.uvws);

    return this.renderer.addFixedMesh(geo);
  }

  private copyInt32s(dst: Uint32Array, src: number[]) {
    for (let i = 0; i < dst.length; i++) dst[i] = src[i];
  }

  private copyFloats(dst: Float32Array, src: number[]) {
    for (let i = 0; i < dst.length; i++) dst[i] = src[i];
  }

  private computeGeometryData(voxels: Tensor3): GeometryData {
    const result = kGeometryData;
    result.numQuads = 0;

    const {data, shape, stride} = voxels;

    for (let d = 0; d < 3; d++) {
      const dir = d * 2;
      const u = (d + 1) % 3;
      const v = (d + 2) % 3;
      const ld = shape[d] - 2,  lu = shape[u] - 2,  lv = shape[v] - 2;
      const sd = stride[d], su = stride[u], sv = stride[v];
      const base = su + sv;

      Vec3.set(kTmpPos,    0, 0, 0);

      const area = lu * lv;
      if (kMaskData.length < area) {
        kMaskData = new Int16Array(area);
      }

      for (let id = 0; id < ld; id++) {
        let n = 0;
        for (let iu = 0; iu < lu; iu++) {
          let index = base + id * sd + iu * su;
          for (let iv = 0; iv < lv; iv++, index += sv, n += 1) {
            // mask[n] is the face between (id, iu, iv) and (id + 1, iu, iv).
            // Its value is the MaterialId to use, times -1, if it is in the
            // direction opposite `dir`.
            //
            // When we enable ambient occlusion, we shift these masks left by
            // 8 bits and pack AO values for each vertex into the lower byte.
            const block0 = data[index] as BlockId;
            const block1 = data[index + sd] as BlockId;
            if (block0 === block1) continue;
            const facing = this.getFaceDir(block0, block1, dir);
            if (facing === 0) continue;

            const mask = facing > 0
              ?  this.getBlockFaceMaterial(block0, dir)
              : -this.getBlockFaceMaterial(block1, dir + 1);
            const ao = facing > 0
              ? this.packAOMask(data, index + sd, index, su, sv)
              : this.packAOMask(data, index, index + sd, su, sv)
            kMaskData[n] = (mask << 8) | ao;
          }
        }

        n = 0;
        kTmpPos[d] = id;

        for (let iu = 0; iu < lu; iu++) {
          let h = 1;
          for (let iv = 0; iv < lv; iv += h, n += h) {
            const mask = kMaskData[n];
            if (mask === 0) {
              h = 1;
              continue;
            }

            for (h = 1; h < lv - iv; h++) {
              if (mask != kMaskData[n + h]) break;
            }

            let w = 1, nw = n + lv;
            OUTER:
            for (; w < lu - iu; w++, nw += lv) {
              for (let x = 0; x < h; x++) {
                if (mask != kMaskData[nw + x]) break OUTER;
              }
            }

            kTmpPos[u] = iu;
            kTmpPos[v] = iv;
            this.addQuad(result, d, u, v, w, h, mask, kTmpPos);

            nw = n;
            for (let wx = 0; wx < w; wx++, nw += lv) {
              for (let hx = 0; hx < h; hx++) {
                kMaskData[nw + hx] = 0;
              }
            }
          }
        }
      }
    }

    return result;
  }

  private addQuad(geo: GeometryData, d: int, u: int, v: int,
                  w: int, h: int, mask: int, pos: Vec3) {
    const {numQuads, positions, normals, indices, colors, uvws} = geo;
    geo.numQuads++;

    const positions_offset = numQuads * 12;
    const indices_offset   = numQuads * 6;
    const colors_offset    = numQuads * 16;
    const base_index       = numQuads * 4;

    if (positions.length < positions_offset + 12) {
      for (let i = 0; i < 12; i++) positions.push(0);
      for (let i = 0; i < 12; i++) normals.push(0);
      for (let i = 0; i < 6; i++)  indices.push(0);
      for (let i = 0; i < 16; i++) colors.push(0);
      for (let i = 0; i < 12; i++)  uvws.push(0);
    }

    const dir = Math.sign(mask);
    for (let i = 0; i < 3; i++) {
      const p = pos[i];
      positions[positions_offset + i + 0] = p;
      positions[positions_offset + i + 3] = p;
      positions[positions_offset + i + 6] = p;
      positions[positions_offset + i + 9] = p;

      const x = i === d ? dir : 0;
      normals[positions_offset + i + 0] = x;
      normals[positions_offset + i + 3] = x;
      normals[positions_offset + i + 6] = x;
      normals[positions_offset + i + 9] = x;
    }
    positions[positions_offset + u + 3] += w;
    positions[positions_offset + u + 6] += w;
    positions[positions_offset + v + 6] += h;
    positions[positions_offset + v + 9] += h;

    const triangleHint = this.getTriangleHint(mask);
    const offsets = mask > 0
      ? (triangleHint ? kIndexOffsets.C : kIndexOffsets.D)
      : (triangleHint ? kIndexOffsets.A : kIndexOffsets.B);
    for (let i = 0; i < 6; i++) {
      indices[indices_offset + i] = base_index + offsets[i];
    }

    const id = Math.abs(mask >> 8) as MaterialId;
    const material = this.getMaterialData(id);
    let textureIndex = material.textureIndex;
    if (textureIndex === 0 && material.texture) {
      textureIndex = this.renderer.atlas.addImage(material.texture);
      material.textureIndex = textureIndex;
    }

    const color = material.color;
    for (let i = 0; i < 4; i++) {
      const ao = 1 - 0.3 * (mask >> (2 * i) & 3);
      colors[colors_offset + 4 * i + 0] = color[0] * ao;
      colors[colors_offset + 4 * i + 1] = color[1] * ao;
      colors[colors_offset + 4 * i + 2] = color[2] * ao;
      colors[colors_offset + 4 * i + 3] = color[3];
    }

    for (let i = 0; i < 12; i++) uvws[positions_offset + i] = 0;
    if (d === 2) {
      uvws[positions_offset + 1] = uvws[positions_offset + 4] = h;
      uvws[positions_offset + 3] = uvws[positions_offset + 6] = -dir * w;
    } else {
      uvws[positions_offset + 1] = uvws[positions_offset + 10] = w;
      uvws[positions_offset + 6] = uvws[positions_offset + 9] = dir * h;
    }
    for (let i = 0; i < 4; i++) {
      uvws[positions_offset + i * 3 + 2] = textureIndex;
    }
  }

  private getFaceDir(block0: BlockId, block1: BlockId, dir: int) {
    const opaque0 = this.opaque[block0];
    const opaque1 = this.opaque[block1];
    if (opaque0 && opaque1) return 0;
    if (opaque0) return 1;
    if (opaque1) return -1;

    const material0 = this.getBlockFaceMaterial(block0, dir);
    const material1 = this.getBlockFaceMaterial(block1, dir + 1);
    if (material0 === material1) return 0;
    if (material0 === kNoMaterial) return -1;
    if (material1 === kNoMaterial) return 1;
    return 0;
  }

  private getTriangleHint(mask: int): boolean {
    const a00 = (mask >> 0) & 3;
    const a10 = (mask >> 2) & 3;
    const a11 = (mask >> 4) & 3;
    const a01 = (mask >> 6) & 3;
    if (a00 === a11) return (a10 === a01) ? a10 === 3 : true;
    return (a10 === a01) ? false : (a00 + a11 > a10 + a01);
  }

  private packAOMask(data: Uint32Array, ipos: int, ineg: int,
                     dj: int, dk: int): int {
    let a00 = 0; let a01 = 0; let a10 = 0; let a11 = 0;
    if (this.solid[data[ipos + dj]]) { a10++; a11++; }
    if (this.solid[data[ipos - dj]]) { a00++; a01++; }
    if (this.solid[data[ipos + dk]]) { a01++; a11++; }
    if (this.solid[data[ipos - dk]]) { a00++; a10++; }

    if (a00 === 0 && this.solid[data[ipos - dj - dk]]) a00++;
    if (a01 === 0 && this.solid[data[ipos - dj + dk]]) a01++;
    if (a10 === 0 && this.solid[data[ipos + dj - dk]]) a10++;
    if (a11 === 0 && this.solid[data[ipos + dj + dk]]) a11++;

    // Order here matches the order in which we push vertices in addQuad.
    return (a01 << 6) | (a11 << 4) | (a10 << 2) | a00;
  }
};

//////////////////////////////////////////////////////////////////////////////

export {TerrainMesher};
