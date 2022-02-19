const nonnull = <T>(x: T | null, message?: () => string): T => {
  if (x !== null) return x;
  throw new Error(message ? message() : 'Unexpected null!');
};

class Container {
  _element: Element;
  _canvas: Element;

  constructor(id: string) {
    this._element = nonnull(document.getElementById(id), () => id);
    this._canvas = nonnull(this._element.querySelector('canvas'));
  }
};

const main = () => {
  const container = new Container('container');
};

window.onload = main;
