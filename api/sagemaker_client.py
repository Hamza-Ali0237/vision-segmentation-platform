"""
SageMaker endpoint client.

Handles:
  - Calling the correct endpoint for each architecture
  - Serialising the image for the request
  - Deserialising the JSON response from the endpoint
  - Graceful error handling with descriptive messages
"""

import json
import boto3
import numpy as np
from botocore.exceptions import ClientError


class SageMakerClient:
    """
    Thin wrapper around boto3's sagemaker-runtime client.

    Endpoint names follow the convention set in aws/deploy.py:
      cxr-seg-{arch}-endpoint

    Override via the ENDPOINT_UNET / ENDPOINT_SEGNET env vars if you've
    used custom names.
    """

    def __init__(self, region: str = "us-east-1"):
        self.runtime = boto3.client("sagemaker-runtime", region_name=region)
        self._endpoint_map = {
            "unet":   "cxr-seg-unet-endpoint",
            "segnet": "cxr-seg-segnet-endpoint",
        }

    def set_endpoint(self, arch: str, endpoint_name: str):
        """Override the default endpoint name for a given architecture."""
        self._endpoint_map[arch] = endpoint_name

    def get_endpoint_name(self, arch: str) -> str:
        if arch not in self._endpoint_map:
            raise ValueError(
                f"Unknown architecture '{arch}'. Choose from: {list(self._endpoint_map)}"
            )
        return self._endpoint_map[arch]

    def predict(
        self,
        image_bytes: bytes,
        arch: str = "unet",
        content_type: str = "application/octet-stream",
    ) -> dict:
        """
        Send raw image bytes to the SageMaker endpoint and return the
        parsed prediction dict with keys 'mask' and 'probs'.

        Raises:
            ValueError  — unknown arch or endpoint not found
            RuntimeError — SageMaker call failed
        """
        endpoint_name = self.get_endpoint_name(arch)

        try:
            response = self.runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType=content_type,
                Body=image_bytes,
            )
        except ClientError as e:
            code = e.response["Error"]["Code"]
            msg  = e.response["Error"]["Message"]
            if code == "ValidationError" and "Could not find endpoint" in msg:
                raise RuntimeError(
                    f"Endpoint '{endpoint_name}' not found. "
                    f"Run 'python aws/deploy.py --arch {arch}' first."
                ) from e
            raise RuntimeError(f"SageMaker error [{code}]: {msg}") from e

        result = json.loads(response["Body"].read())
        return {
            "mask":  np.array(result["mask"],  dtype=np.int32),
            "probs": np.array(result["probs"], dtype=np.float32),
        }

    def list_available_endpoints(self) -> list[str]:
        """
        Return the list of architecture names whose endpoints currently
        exist in SageMaker (i.e. are deployed and in service).
        """
        sm = boto3.client(
            "sagemaker",
            region_name=self.runtime.meta.region_name,
        )
        available = []
        for arch, endpoint_name in self._endpoint_map.items():
            try:
                resp = sm.describe_endpoint(EndpointName=endpoint_name)
                if resp["EndpointStatus"] == "InService":
                    available.append(arch)
            except ClientError:
                pass
        return available