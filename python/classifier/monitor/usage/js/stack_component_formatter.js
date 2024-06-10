const i = special_vars.index;
const glyph = special_vars.glyph_view;
const height = glyph.y2[i] - glyph.y1[i];
return height.toFixed(2) + "{{ unit }}";