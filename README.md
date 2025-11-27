# About 

This repository you can find the code for running [WhisperX](https://github.com/m-bain/whisperX) using [KServe Model Server API](https://kserve.github.io/website/docs/model-serving/predictive-inference/frameworks/custom-predictor).


> [!WARNING]  
> Use it as your own risk. The code of the service - `whisperx_predictor.py` is generated using vibe coding


# How to use

First build the image

```
podman build . quay.io/demo-org/whisperx-kserve:v0.1
```

And push it to a container registry

```
podman push quay.io/demo-org/whisperx-kserve:v0.1
```

Once the image is created, you can deploy it on OpenShift AI

```
oc apply -f manifests/infereceservice.yaml
```

Once the pod is up and runnning, test the deployment and the model

```
AUDIO_B64=$(base64 -i test.wav | tr -d '\n')

curl -X POST https://<openshift-route-whisperx-predictor>/v1/models/whisperx:predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d "{
    \"instances\": [{
      \"audio\": \"$AUDIO_B64\",
      \"align\": true,
      \"diarize\": false
    }]
  }"
```

