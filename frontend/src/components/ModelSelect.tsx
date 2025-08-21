"use client";

import React from "react";
import { Box, Text, NativeSelect } from "@chakra-ui/react";

interface ModelSelectProps {
  modelId: "base" | "fine_tuned";
  onChange: (value: "base" | "fine_tuned") => void;
}

export default function ModelSelect({ modelId, onChange }: ModelSelectProps) {
  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.currentTarget.value as "base" | "fine_tuned";
    onChange(value);
  };

  return (
    <Box>
      <Text mb="2" fontWeight="medium">
        Model
      </Text>
      <NativeSelect.Root size="sm" width="280px">
        <NativeSelect.Field
          value={modelId}
          onChange={handleChange}
          aria-label="Select ASR model"
        >
          <option value="base">whisper_tiny_de</option>
          <option value="fine_tuned">whisper_tiny_de_finetuned</option>
        </NativeSelect.Field>
        <NativeSelect.Indicator />
      </NativeSelect.Root>
    </Box>
  );
}