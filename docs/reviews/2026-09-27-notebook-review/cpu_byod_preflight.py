"""BYOD pre-flight: a ZIP built from the cached real photos, run through the notebook's BYOD cell with two real backbones."""
import contextlib, io, json, os, sys, zipfile
from pathlib import Path
os.environ["MPLBACKEND"]="Agg"
nb, workdir, out = sys.argv[1:4]
cells={c["id"]:"".join(c["source"]) for c in json.loads(Path(nb).read_text())["cells"] if c["cell_type"]=="code"}
os.chdir(workdir)
ns={"display":print}
for cid in ["4f54b044","aa91b536","e04f3109","1b442017","f470276e","f370927c"]:
    with contextlib.redirect_stdout(io.StringIO()):
        exec(cells[cid],ns)
exec(cells["6e025182"][:cells["6e025182"].index("ALL_RESULTS={}")],ns)
exec(cells["22e6f1e2"][:cells["22e6f1e2"].index("model_metric_rows=[]")],ns)
names={"song_sparrow":"001","chipping_sparrow":"NA","white_throated_sparrow":"0","dark_eyed_junco":"junco — "+"é"*140,
       "house_finch":"house finch","american_goldfinch":"American Goldfinch"}
archive=Path(workdir)/"byod_birds.zip"
rows=["filename,label,split"]
with zipfile.ZipFile(archive,"w") as z:
    for species in ns["CLASS_KEYS"]:
        recs=[r for r in ns["records"] if r["label"]==species]
        for i,r in enumerate(recs):
            split="train" if i<20 else "validation" if i<25 else "test"
            name=f"birds/{species}/{r['photo_id']}.jpg"
            buf=io.BytesIO(); r["image"].save(buf,format="JPEG",quality=95)
            z.writestr(name,buf.getvalue())
            rows.append(f"birds/{species}/{r['photo_id']}.jpg,{names[species]},{split}")
    z.writestr("labels.csv","\n".join(rows)+"\n")
src=cells["71cf319a"].replace("if USE_BYOD:","if True:")
ns["BYOD_PATH"]=str(archive); ns["BYOD_MODEL_KEYS"]=["mobilenetv4","vit"]
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(src,ns)
root=ns["OUT_ROOT"]/"byod"
res={"stdout":buf.getvalue()[-4000:],"summary":ns["byod_summary"],
     "files":sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()),
     "class_order":json.loads((root/"probes/vit/manifest.json").read_text())["class_order"],
     "predictions_header":(root/"predictions.csv").read_text(encoding="utf-8").splitlines()[0]}
Path(out).write_text(json.dumps(res,indent=2,ensure_ascii=False,default=str))
print(json.dumps(res["summary"],indent=1,default=str))
