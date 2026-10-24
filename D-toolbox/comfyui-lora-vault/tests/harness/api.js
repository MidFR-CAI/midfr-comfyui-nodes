const L = {};
export const api = { fetchApi: (p, init) => fetch(p, init), addEventListener(n, f) { L[n] = f; }, _emit(n, d) { L[n] && L[n]({ detail: d }); } };
